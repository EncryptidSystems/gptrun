// Package rt talks to dawnrt. Buffers and pipelines stay on the device.
// A Run is one submit: the weights are not written again.
package rt

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

const (
	opShader = 1
	opBuf    = 2
	opPipe   = 3
	opGroup  = 4
	opRun    = 5
	opWrite  = 7
	opInfo   = 8
	opRun2   = 9

	KindRO  = 0
	KindRW  = 1
	KindUni = 2
	// KindUniDyn is a uniform binding whose offset each dispatch sets (Op.Dyn).
	KindUniDyn = 3

	OpDispatch = 1
	OpCopy     = 2
	opDispDyn  = 3

	// Info flags.
	InfoDot4      = 1 << 0 // WGSL packed_4x8_integer_dot_product
	InfoSubgroups = 1 << 1
	InfoF16       = 1 << 2
	InfoFast      = 1 << 3 // robustness and workgroup zero-init are off
)

// Exe is dawnrt.exe: MODELRUN_DAWNRT, else beside gptrun.exe (dawnrt\ or the
// same folder, as a release unpacks), else ~/ModelRun/dawnrt.
func Exe() string {
	if v := os.Getenv("MODELRUN_DAWNRT"); v != "" {
		return v
	}
	if self, err := os.Executable(); err == nil {
		dir := filepath.Dir(self)
		for _, p := range []string{filepath.Join(dir, "dawnrt", "dawnrt.exe"), filepath.Join(dir, "dawnrt.exe")} {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "ModelRun", "dawnrt", "dawnrt.exe")
}

// Dev is one device. Ids are the host's.
type Dev struct {
	mu   sync.Mutex
	cmd  *exec.Cmd
	// dead is set when the process stopped or stopped answering; every later
	// call returns it. hung marks a stop by the reply watchdog.
	dead error
	hung atomic.Bool
	in   io.WriteCloser
	out  io.ReadCloser
	wbuf  []byte
	rbody []byte
	hdr   [8]byte
	rbuf  []byte
}

// Open starts the runtime.
func Open() (*Dev, error) {
	exe := Exe()
	if _, err := os.Stat(exe); err != nil {
		return nil, fmt.Errorf("dawnrt %s: %w", exe, err)
	}
	cmd := exec.Command(exe)
	cmd.Stderr = os.Stderr
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &Dev{cmd: cmd, in: in, out: out}, nil
}

// Close stops the process.
func (d *Dev) Close() {
	if d == nil || d.in == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.dead == nil {
		var n uint32 = 4
		_ = binary.Write(d.in, binary.LittleEndian, n)
		_, _ = d.in.Write([]byte("quit"))
	}
	_ = d.in.Close()
	d.in = nil
	if d.cmd != nil {
		_ = d.cmd.Wait()
	}
}

// call sends one command in one write and reads the reply. The returned slice
// is valid until the next call on d.
func (d *Dev) call(parts ...[]byte) ([]byte, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.exchange(func() error {
		n := 0
		for _, p := range parts {
			n += len(p)
		}
		// A large upload goes out as its own writes; copying 2 GB to join it saves nothing.
		if n > 1<<24 {
			var h [4]byte
			binary.LittleEndian.PutUint32(h[:], uint32(n))
			if _, err := d.in.Write(h[:]); err != nil {
				return err
			}
			for _, p := range parts {
				if _, err := d.in.Write(p); err != nil {
					return err
				}
			}
			return nil
		}
		w := d.wbuf[:0]
		w = binary.LittleEndian.AppendUint32(w, uint32(n))
		for _, p := range parts {
			w = append(w, p...)
		}
		d.wbuf = w
		_, err := d.in.Write(w)
		return err
	})
}

// ErrDead is a device whose process stopped or stopped answering. The
// device's buffers and pipelines are gone with it.
var ErrDead = errors.New("the GPU process stopped")

// DevError is an error the device reported for one command. The device
// itself is still up.
type DevError struct{ Msg string }

func (e *DevError) Error() string { return e.Msg }

// ReplyLimit is how long one command may wait for its reply before the
// process is taken for hung and stopped. The slowest is a 2 GB upload.
var ReplyLimit = 3 * time.Minute

// exchange sends one command and reads its reply, with d.mu held. A failed
// write or read is the process gone: d is dead from then on.
func (d *Dev) exchange(send func() error) ([]byte, error) {
	if d.dead != nil {
		return nil, d.dead
	}
	watch := time.AfterFunc(ReplyLimit, func() {
		d.hung.Store(true)
		d.kill()
	})
	err := send()
	var b []byte
	if err == nil {
		b, err = d.readReply()
	}
	watch.Stop()
	var dev *DevError
	if err == nil || errors.As(err, &dev) {
		return b, err
	}
	if d.hung.Load() {
		err = fmt.Errorf("no reply in %v", ReplyLimit)
	}
	d.dead = fmt.Errorf("%w (%v)", ErrDead, err)
	d.kill()
	return nil, d.dead
}

// Err is the reason d stopped, or nil while it is up.
func (d *Dev) Err() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.dead
}

// Kill stops the process now. The next call on d fails with ErrDead. It is
// what a crashed or lost device looks like, for tests and for the watchdog.
func (d *Dev) Kill() { d.kill() }

func (d *Dev) kill() {
	if d.cmd != nil && d.cmd.Process != nil {
		_ = d.cmd.Process.Kill()
	}
}

func (d *Dev) readReply() ([]byte, error) {
	if _, err := io.ReadFull(d.out, d.hdr[:]); err != nil {
		return nil, err
	}
	rc := binary.LittleEndian.Uint32(d.hdr[0:])
	m := int(binary.LittleEndian.Uint32(d.hdr[4:]))
	if cap(d.rbuf) < m {
		d.rbuf = make([]byte, m)
	}
	buf := d.rbuf[:m]
	if _, err := io.ReadFull(d.out, buf); err != nil {
		return nil, err
	}
	if rc != 0 {
		return nil, &DevError{Msg: string(buf)}
	}
	return buf, nil
}

func idOf(b []byte) (uint32, error) {
	if len(b) < 4 {
		return 0, fmt.Errorf("dawnrt: short id")
	}
	return binary.LittleEndian.Uint32(b), nil
}

func u32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}

func u64(v uint64) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, v)
	return b
}

func str(s string) []byte {
	b := u32(uint32(len(s)))
	return append(b, s...)
}

// Shader compiles a WGSL module.
func (d *Dev) Shader(name, src string) error {
	_, err := d.call(u32(opShader), str(name), u32(uint32(len(src))), []byte(src))
	return err
}

// Buf creates a buffer and writes data at the start. size 0 means len(data).
func (d *Dev) Buf(name string, data []byte, size int) (uint32, error) {
	if size < len(data) {
		size = len(data)
	}
	b, err := d.call(u32(opBuf), str(name), u64(uint64(size)), u64(uint64(len(data))), data)
	if err != nil {
		return 0, err
	}
	return idOf(b)
}

// Write replaces bytes in an existing buffer.
func (d *Dev) Write(name string, off int, data []byte) error {
	_, err := d.call(u32(opWrite), str(name), u32(uint32(off)), u32(uint32(len(data))), data)
	return err
}

// Pipe compiles one entry. kinds are KindRO, KindRW, KindUni in binding order.
func (d *Dev) Pipe(name, shader, entry, overrides string, kinds []byte) (uint32, error) {
	parts := [][]byte{u32(opPipe), str(name), str(shader), str(entry), str(overrides), u32(uint32(len(kinds))), kinds}
	b, err := d.call(parts...)
	if err != nil {
		return 0, err
	}
	return idOf(b)
}

// Bind is one binding: buffer name, byte offset, byte size. Size 0 is the rest.
type Bind struct {
	Buf  string
	Off  int
	Size int
}

// Group binds buffers to a pipeline. The order is the binding order.
func (d *Dev) Group(name, pipe string, binds []Bind) (uint32, error) {
	parts := [][]byte{u32(opGroup), str(name), str(pipe), u32(uint32(len(binds)))}
	for _, b := range binds {
		parts = append(parts, str(b.Buf), u32(uint32(b.Off)), u32(uint32(b.Size)))
	}
	out, err := d.call(parts...)
	if err != nil {
		return 0, err
	}
	return idOf(out)
}

// Op is one recorded command. Tag OpDispatch uses Pipe, Group, X, Y, Z, and
// Dyn, the byte offset of the pipe's KindUniDyn binding.
// Tag OpCopy uses Src, SrcOff, Dst, DstOff, N.
type Op struct {
	Tag            byte
	Pipe, Group    uint32
	X, Y, Z        uint32
	Dyn            uint32
	Src, Dst       uint32
	SrcOff, DstOff uint32
	N              uint32
}

// Get reads bytes after the submit.
type Get struct {
	Buf uint32
	Off int
	N   int
}

// Run submits ops as one command buffer, then reads gets. It waits for the device.
func (d *Dev) Run(ops []Op, gets []Get) ([]byte, error) {
	out, err := d.Run2(nil, ops, gets, true)
	if err != nil {
		return nil, err
	}
	return append([]byte(nil), out...), nil
}

// Write is one queue write done at the start of a Run2, before its ops.
type Write struct {
	Buf  uint32
	Off  int
	Data []byte
}

// Run2 writes, records ops into one command buffer, copies gets out in that same
// command buffer, and submits once. With no gets and wait false it returns when
// the runtime has submitted, not when the device is done: the queue orders the
// next run after it. The returned bytes are valid until the next call on d.
func (d *Dev) Run2(writes []Write, ops []Op, gets []Get, wait bool) ([]byte, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.exchange(func() error {
		_, err := d.in.Write(d.encodeRun2(writes, ops, gets, wait))
		return err
	})
}

// encodeRun2 builds the length-prefixed command in a buffer it keeps for the
// next run. d.mu is held.
func (d *Dev) encodeRun2(writes []Write, ops []Op, gets []Get, wait bool) []byte {
	b := d.rbody[:0]
	le := binary.LittleEndian
	b = le.AppendUint32(b, 0) // length, set below
	b = le.AppendUint32(b, opRun2)
	flags := uint32(0)
	if wait {
		flags |= 1
	}
	b = le.AppendUint32(b, flags)
	b = le.AppendUint32(b, uint32(len(writes)))
	for _, w := range writes {
		b = le.AppendUint32(b, w.Buf)
		b = le.AppendUint32(b, uint32(w.Off))
		b = le.AppendUint32(b, uint32(len(w.Data)))
		b = append(b, w.Data...)
	}
	b = le.AppendUint32(b, uint32(len(ops)))
	for _, op := range ops {
		b = appendOp(b, op)
	}
	b = le.AppendUint32(b, uint32(len(gets)))
	for _, g := range gets {
		b = le.AppendUint32(b, g.Buf)
		b = le.AppendUint32(b, uint32(g.Off))
		b = le.AppendUint32(b, uint32(g.N))
	}
	le.PutUint32(b[0:], uint32(len(b)-4))
	d.rbody = b
	return b
}

func appendOp(b []byte, op Op) []byte {
	le := binary.LittleEndian
	if op.Tag == OpCopy {
		b = append(b, OpCopy)
		b = le.AppendUint32(b, op.Src)
		b = le.AppendUint32(b, op.SrcOff)
		b = le.AppendUint32(b, op.Dst)
		b = le.AppendUint32(b, op.DstOff)
		return le.AppendUint32(b, op.N)
	}
	tag := byte(OpDispatch)
	if op.Dyn != 0 {
		tag = opDispDyn
	}
	b = append(b, tag)
	b = le.AppendUint32(b, op.Pipe)
	b = le.AppendUint32(b, op.Group)
	b = le.AppendUint32(b, op.X)
	b = le.AppendUint32(b, op.Y)
	b = le.AppendUint32(b, op.Z)
	if tag == opDispDyn {
		b = le.AppendUint32(b, op.Dyn)
	}
	return b
}

// Info is the device's flags (InfoDot4, ...) and its uniform offset alignment.
// An older dawnrt has no info op and no Run2: that is an error.
func (d *Dev) Info() (flags uint32, uniAlign int, err error) {
	b, err := d.call(u32(opInfo))
	if err != nil {
		return 0, 0, err
	}
	if len(b) < 8 {
		return 0, 0, fmt.Errorf("dawnrt: short info")
	}
	return binary.LittleEndian.Uint32(b), int(binary.LittleEndian.Uint32(b[4:])), nil
}
