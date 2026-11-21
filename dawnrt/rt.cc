// Dawn runtime. Weights stay in their buffers. A run is one command buffer:
// queue writes, then compute passes and copies, then one submit. Readback is
// copied into a resident staging buffer inside that same command buffer, so a
// token is one submit and one wait. Waits block on the device fence
// (wgpuInstanceWaitAny), not on a Sleep(1) poll.
#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <io.h>
#include <windows.h>

#pragma comment(lib, "winmm.lib")
extern "C" __declspec(dllimport) unsigned int __stdcall timeBeginPeriod(unsigned int);

static WGPUStringView Sv(const char* s) {
  WGPUStringView v = WGPU_STRING_VIEW_INIT;
  if (!s) return v;
  v.data = s;
  v.length = std::strlen(s);
  return v;
}

static WGPUStringView Sv(const std::string& s) {
  WGPUStringView v = WGPU_STRING_VIEW_INIT;
  v.data = s.data();
  v.length = s.size();
  return v;
}

static std::string ToStr(WGPUStringView v) {
  if (!v.data || v.length == 0) return {};
  size_t n = v.length == WGPU_STRLEN ? std::strlen(v.data) : v.length;
  return std::string(v.data, n);
}

struct Buf {
  WGPUBuffer b = nullptr;
  uint64_t size = 0;
  std::string name;
};

struct Pipe {
  WGPUComputePipeline pipe = nullptr;
  WGPUPipelineLayout layout = nullptr;
  WGPUBindGroupLayout bgl = nullptr;
  std::string name;
  std::vector<std::string> keys;
  int nbind = 0;
  int ndyn = 0;  // uniform bindings with a dynamic offset
};

struct Group {
  WGPUBindGroup g = nullptr;
  std::string name;
  uint32_t pipe = 0;
};

// Info flags (op 8).
enum : uint32_t {
  kInfoDot4 = 1u << 0,      // WGSL packed_4x8_integer_dot_product
  kInfoSubgroups = 1u << 1,
  kInfoF16 = 1u << 2,
  kInfoFast = 1u << 3,      // robustness and workgroup zero-init are off
};

struct Dev {
  WGPUInstance instance = nullptr;
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  std::string err;
  std::vector<WGPUShaderModule> shaders;
  std::map<std::string, uint32_t> shaderID;
  std::vector<Buf> bufs;
  std::map<std::string, uint32_t> bufID;
  std::vector<Pipe> pipes;
  std::map<std::string, uint32_t> pipeID;
  std::vector<Group> groups;
  std::map<std::string, uint32_t> groupID;
  WGPUBuffer staging = nullptr;
  uint64_t stagingSize = 0;
  uint32_t info = 0;
  uint32_t uniAlign = 256;
  bool timed = false;
};

static Dev g;
static bool g_time = false;  // MODELRUN_DAWN_TIME: per-run timing on stderr

static double NowMs() {
  static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return 1000.0 * static_cast<double>(t.QuadPart) / static_cast<double>(f.QuadPart);
}

static void OnUncaptured(WGPUDevice const*, WGPUErrorType, WGPUStringView message, void*, void*) {
  g.err += "dawn: " + ToStr(message) + "\n";
}

// Pump is the setup path's wait (adapter, device, shader info). It is not on the token path.
static void Pump(bool* done) {
  for (int i = 0; i < 200000 && !*done; ++i) {
    wgpuInstanceProcessEvents(g.instance);
    if (!*done) Sleep(i < 64 ? 0 : 1);
  }
}

// Wait blocks on one future. Without timed waits it spins on ProcessEvents with a yield.
static void Wait(WGPUFuture f, bool* done) {
  if (g.timed) {
    WGPUFutureWaitInfo w = WGPU_FUTURE_WAIT_INFO_INIT;
    w.future = f;
    while (!*done) {
      WGPUWaitStatus s = wgpuInstanceWaitAny(g.instance, 1, &w, UINT64_MAX);
      if (s != WGPUWaitStatus_Success) break;
      if (w.completed) break;
    }
    if (*done) return;
  }
  while (!*done) {
    wgpuInstanceProcessEvents(g.instance);
    if (!*done) SwitchToThread();
  }
}

static bool EnvOn(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::strcmp(v, "0") != 0;
}

static bool Open(std::string* err) {
  WGPUInstanceDescriptor id = WGPU_INSTANCE_DESCRIPTOR_INIT;
  id.capabilities.timedWaitAnyEnable = 1;
  id.capabilities.timedWaitAnyMaxCount = 1;
  g.instance = wgpuCreateInstance(&id);
  g.timed = g.instance != nullptr;
  if (!g.instance) {
    WGPUInstanceDescriptor plain = WGPU_INSTANCE_DESCRIPTOR_INIT;
    g.instance = wgpuCreateInstance(&plain);
  }
  if (!g.instance) {
    *err = "dawn: wgpuCreateInstance failed\n";
    return false;
  }
  struct Ad {
    WGPUAdapter adapter = nullptr;
    bool done = false;
    std::string msg;
  } ad;
  WGPURequestAdapterOptions opts = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
  opts.featureLevel = WGPUFeatureLevel_Core;
  opts.powerPreference = WGPUPowerPreference_HighPerformance;
  WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
  acb.mode = WGPUCallbackMode_AllowSpontaneous;
  acb.userdata1 = &ad;
  acb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* u1, void*) {
    Ad& a = *static_cast<Ad*>(u1);
    if (status == WGPURequestAdapterStatus_Success) a.adapter = adapter;
    else a.msg = ToStr(message);
    a.done = true;
  };
  wgpuInstanceRequestAdapter(g.instance, &opts, acb);
  if (!ad.done) Pump(&ad.done);
  if (!ad.adapter) {
    *err = "dawn: no adapter" + (ad.msg.empty() ? "" : ": " + ad.msg) + "\n";
    return false;
  }
  g.adapter = ad.adapter;
  WGPULimits limits = WGPU_LIMITS_INIT;
  bool have = wgpuAdapterGetLimits(g.adapter, &limits) == WGPUStatus_Success;
  if (have && limits.minUniformBufferOffsetAlignment) g.uniAlign = limits.minUniformBufferOffsetAlignment;

  std::vector<WGPUFeatureName> feats;
  if (wgpuAdapterHasFeature(g.adapter, WGPUFeatureName_Subgroups)) {
    feats.push_back(WGPUFeatureName_Subgroups);
    g.info |= kInfoSubgroups;
  }
  if (wgpuAdapterHasFeature(g.adapter, WGPUFeatureName_ShaderF16)) {
    feats.push_back(WGPUFeatureName_ShaderF16);
    g.info |= kInfoF16;
  }
  if (wgpuInstanceHasWGSLLanguageFeature(g.instance, WGPUWGSLLanguageFeatureName_Packed4x8IntegerDotProduct)) {
    g.info |= kInfoDot4;
  }

  // Every kernel writes its workgroup memory before reading it and stays in
  // bounds, so Dawn's zero-init and clamping are pure cost. MODELRUN_DAWN_SAFE=1
  // turns them back on.
  const bool safe = EnvOn("MODELRUN_DAWN_SAFE");
  static const char* fast[] = {"disable_robustness", "disable_workgroup_init", "skip_validation"};
  WGPUDawnTogglesDescriptor toggles = WGPU_DAWN_TOGGLES_DESCRIPTOR_INIT;
  toggles.enabledToggleCount = safe ? 0 : sizeof(fast) / sizeof(fast[0]);
  toggles.enabledToggles = fast;
  if (!safe) g.info |= kInfoFast;

  struct Dd {
    WGPUDevice device = nullptr;
    bool done = false;
    std::string msg;
  } dd;
  WGPUDeviceDescriptor desc = WGPU_DEVICE_DESCRIPTOR_INIT;
  desc.nextInChain = &toggles.chain;
  desc.label = Sv("ModelRun Dawn");
  desc.requiredLimits = have ? &limits : nullptr;
  desc.requiredFeatureCount = feats.size();
  desc.requiredFeatures = feats.empty() ? nullptr : feats.data();
  desc.uncapturedErrorCallbackInfo.callback = OnUncaptured;
  desc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
  // A lost device (a driver reset, sleep, a driver update) cannot run another
  // command. Exiting turns it into a closed pipe, which ModelRun recovers from
  // by starting a new dawnrt and loading the model again. A device released
  // at exit is not a loss.
  desc.deviceLostCallbackInfo.callback = [](WGPUDevice const*, WGPUDeviceLostReason reason, WGPUStringView message, void*, void*) {
    if (reason == WGPUDeviceLostReason_Destroyed || reason == WGPUDeviceLostReason_CallbackCancelled) return;
    int n = message.data == nullptr ? 0 : (message.length == WGPU_STRLEN ? (int)strlen(message.data) : (int)message.length);
    fprintf(stderr, "dawnrt: device lost (reason %d): %.*s\n", (int)reason, n, n ? message.data : "");
    fflush(stderr);
    _exit(3);
  };
  WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
  dcb.mode = WGPUCallbackMode_AllowSpontaneous;
  dcb.userdata1 = &dd;
  dcb.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* u1, void*) {
    Dd& d = *static_cast<Dd*>(u1);
    if (status == WGPURequestDeviceStatus_Success) d.device = device;
    else d.msg = ToStr(message);
    d.done = true;
  };
  wgpuAdapterRequestDevice(g.adapter, &desc, dcb);
  if (!dd.done) Pump(&dd.done);
  if (!dd.device) {
    // Fall back to the defaults: no limits, no features, no toggles.
    dd = Dd();
    desc.requiredLimits = nullptr;
    desc.requiredFeatureCount = 0;
    desc.requiredFeatures = nullptr;
    desc.nextInChain = nullptr;
    g.info &= ~(kInfoSubgroups | kInfoF16 | kInfoFast);
    wgpuAdapterRequestDevice(g.adapter, &desc, dcb);
    if (!dd.done) Pump(&dd.done);
  }
  if (!dd.device) {
    *err = "dawn: no device" + (dd.msg.empty() ? "" : ": " + dd.msg) + "\n";
    return false;
  }
  g.device = dd.device;
  g.queue = wgpuDeviceGetQueue(g.device);
  return true;
}

struct R {
  const uint8_t* p;
  size_t n;
  size_t o = 0;
  bool ok = true;
  bool need(size_t k) {
    if (!ok || o + k > n) {
      ok = false;
      return false;
    }
    return true;
  }
  uint32_t u32() {
    if (!need(4)) return 0;
    uint32_t v;
    std::memcpy(&v, p + o, 4);
    o += 4;
    return v;
  }
  uint64_t u64() {
    if (!need(8)) return 0;
    uint64_t v;
    std::memcpy(&v, p + o, 8);
    o += 8;
    return v;
  }
  uint8_t u8() {
    if (!need(1)) return 0;
    return p[o++];
  }
  std::string str() {
    uint32_t n = u32();
    if (!need(n)) return {};
    std::string s(reinterpret_cast<const char*>(p + o), n);
    o += n;
    return s;
  }
  const uint8_t* bytes(uint64_t n) {
    if (!need(static_cast<size_t>(n))) return nullptr;
    const uint8_t* b = p + o;
    o += static_cast<size_t>(n);
    return b;
  }
};

static bool rd(void* p, size_t n) {
  auto* b = static_cast<unsigned char*>(p);
  size_t o = 0;
  while (o < n) {
    size_t k = std::fread(b + o, 1, n - o, stdin);
    if (k == 0) return false;
    o += k;
  }
  return true;
}

static bool wr(const void* p, size_t n) {
  auto* b = static_cast<const unsigned char*>(p);
  size_t o = 0;
  while (o < n) {
    size_t k = std::fwrite(b + o, 1, n - o, stdout);
    if (k == 0) return false;
    o += k;
  }
  return true;
}

// reply is one write and one flush: the header and the body go out together.
static std::string g_out;

static void replyRaw(uint32_t rc, const char* body, size_t n) {
  g_out.resize(8 + n);
  uint32_t n32 = static_cast<uint32_t>(n);
  std::memcpy(&g_out[0], &rc, 4);
  std::memcpy(&g_out[4], &n32, 4);
  if (n) std::memcpy(&g_out[8], body, n);
  wr(g_out.data(), g_out.size());
  std::fflush(stdout);
}

static void reply(uint32_t rc, const std::string& body) { replyRaw(rc, body.data(), body.size()); }

static void replyID(uint32_t id) { replyRaw(0, reinterpret_cast<const char*>(&id), 4); }

static uint64_t RoundUp(uint64_t a, uint64_t n) { return (n + a - 1) / a * a; }

static bool TakeErr(std::string* err) {
  wgpuInstanceProcessEvents(g.instance);
  if (g.err.empty()) return true;
  *err = g.err;
  g.err.clear();
  return false;
}

static bool DoShader(R& r, std::string* err) {
  std::string name = r.str();
  uint32_t n = r.u32();
  const uint8_t* src = r.bytes(n);
  if (!r.ok || !src) {
    *err = "shader: short read\n";
    return false;
  }
  std::string code(reinterpret_cast<const char*>(src), n);
  WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
  wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl.code = Sv(code);
  WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
  md.nextInChain = &wgsl.chain;
  md.label = Sv(name);
  g.err.clear();
  WGPUShaderModule mod = wgpuDeviceCreateShaderModule(g.device, &md);
  if (!mod) {
    *err = "shader: module was not created\n";
    return false;
  }
  struct Ctx {
    std::string text;
    int errors = 0;
    bool done = false;
  } ctx;
  WGPUCompilationInfoCallbackInfo cb = WGPU_COMPILATION_INFO_CALLBACK_INFO_INIT;
  cb.mode = WGPUCallbackMode_AllowSpontaneous;
  cb.userdata1 = &ctx;
  cb.callback = [](WGPUCompilationInfoRequestStatus, WGPUCompilationInfo const* info, void* u1, void*) {
    Ctx& c = *static_cast<Ctx*>(u1);
    for (size_t i = 0; info && i < info->messageCount; ++i) {
      const WGPUCompilationMessage& m = info->messages[i];
      if (m.type == WGPUCompilationMessageType_Error) {
        ++c.errors;
        c.text += "error: " + ToStr(m.message) + "\n";
      }
    }
    c.done = true;
  };
  wgpuShaderModuleGetCompilationInfo(mod, cb);
  if (!ctx.done) Pump(&ctx.done);
  if (ctx.errors || !TakeErr(err)) {
    if (!ctx.text.empty()) *err += ctx.text;
    wgpuShaderModuleRelease(mod);
    return false;
  }
  auto it = g.shaderID.find(name);
  if (it != g.shaderID.end()) {
    wgpuShaderModuleRelease(g.shaders[it->second - 1]);
    g.shaders[it->second - 1] = mod;
    replyID(it->second);
    return true;
  }
  g.shaders.push_back(mod);
  uint32_t id = static_cast<uint32_t>(g.shaders.size());
  g.shaderID[name] = id;
  replyID(id);
  return true;
}

static bool DoBuf(R& r, std::string* err) {
  std::string name = r.str();
  uint64_t size = r.u64();
  uint64_t dn = r.u64();
  const uint8_t* data = dn ? r.bytes(dn) : nullptr;
  if (!r.ok || (dn && !data)) {
    *err = "buf: short read\n";
    return false;
  }
  if (size < dn) size = dn;
  size = RoundUp(4, size < 16 ? 16 : size);
  auto it = g.bufID.find(name);
  WGPUBuffer buf = nullptr;
  uint32_t id = 0;
  if (it == g.bufID.end()) {
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.label = Sv(name);
    bd.size = size;
    bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Uniform | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    buf = wgpuDeviceCreateBuffer(g.device, &bd);
    if (!buf || !TakeErr(err)) {
      if (err->empty()) *err = "buf: create failed\n";
      return false;
    }
    g.bufs.push_back(Buf{buf, size, name});
    id = static_cast<uint32_t>(g.bufs.size());
    g.bufID[name] = id;
  } else {
    id = it->second;
    buf = g.bufs[id - 1].b;
    if (size > g.bufs[id - 1].size) {
      *err = "buf: " + name + " is smaller than the write\n";
      return false;
    }
  }
  if (dn) {
    uint64_t whole = dn & ~uint64_t(3);
    if (whole) wgpuQueueWriteBuffer(g.queue, buf, 0, data, static_cast<size_t>(whole));
    if (whole != dn) {
      uint8_t tail[4] = {0, 0, 0, 0};
      std::memcpy(tail, data + whole, static_cast<size_t>(dn - whole));
      wgpuQueueWriteBuffer(g.queue, buf, whole, tail, 4);
    }
  }
  replyID(id);
  return true;
}

static bool WriteAt(uint32_t id, uint32_t off, const uint8_t* data, uint32_t dn, std::string* err) {
  if (id == 0 || id > g.bufs.size()) {
    *err = "write: buffer id\n";
    return false;
  }
  Buf& b = g.bufs[id - 1];
  uint32_t padded = static_cast<uint32_t>(RoundUp(4, dn));
  if ((off & 3u) != 0 || static_cast<uint64_t>(off) + padded > b.size) {
    *err = "write: past the end of " + b.name + "\n";
    return false;
  }
  if (!dn) return true;
  if (padded == dn) {
    wgpuQueueWriteBuffer(g.queue, b.b, off, data, dn);
    return true;
  }
  std::vector<uint8_t> tmp(padded, 0);
  std::memcpy(tmp.data(), data, dn);
  wgpuQueueWriteBuffer(g.queue, b.b, off, tmp.data(), tmp.size());
  return true;
}

static bool DoWrite(R& r, std::string* err) {
  std::string name = r.str();
  uint32_t off = r.u32();
  uint32_t dn = r.u32();
  const uint8_t* data = dn ? r.bytes(dn) : nullptr;
  if (!r.ok || (dn && !data)) {
    *err = "write: short read\n";
    return false;
  }
  auto it = g.bufID.find(name);
  if (it == g.bufID.end()) {
    *err = "write: no buffer " + name + "\n";
    return false;
  }
  if (!WriteAt(it->second, off, data, dn, err)) return false;
  replyID(it->second);
  return true;
}

static bool DoPipe(R& r, std::string* err) {
  std::string name = r.str();
  std::string shader = r.str();
  std::string entry = r.str();
  std::string over = r.str();
  uint32_t nb = r.u32();
  std::vector<uint8_t> kinds(nb);
  for (uint32_t i = 0; i < nb; ++i) kinds[i] = r.u8();
  if (!r.ok) {
    *err = "pipe: short read\n";
    return false;
  }
  auto sit = g.shaderID.find(shader);
  if (sit == g.shaderID.end()) {
    *err = "pipe: no shader " + shader + "\n";
    return false;
  }
  int ndyn = 0;
  std::vector<WGPUBindGroupLayoutEntry> entries(nb);
  for (uint32_t i = 0; i < nb; ++i) {
    WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    e.binding = i;
    e.visibility = WGPUShaderStage_Compute;
    if (kinds[i] == 2 || kinds[i] == 3) e.buffer.type = WGPUBufferBindingType_Uniform;
    else if (kinds[i] == 1) e.buffer.type = WGPUBufferBindingType_Storage;
    else e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    if (kinds[i] == 3) {
      e.buffer.hasDynamicOffset = 1;
      ++ndyn;
    }
    entries[i] = e;
  }
  WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
  ld.entryCount = entries.size();
  ld.entries = entries.data();
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(g.device, &ld);
  WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
  pld.bindGroupLayoutCount = 1;
  pld.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g.device, &pld);

  std::vector<std::string> keys;
  std::vector<double> vals;
  std::string cur;
  auto flush = [&]() {
    if (cur.empty()) return;
    auto eq = cur.find('=');
    if (eq == std::string::npos) return;
    keys.push_back(cur.substr(0, eq));
    std::string v = cur.substr(eq + 1);
    vals.push_back(v == "true" ? 1.0 : v == "false" ? 0.0 : std::strtod(v.c_str(), nullptr));
    cur.clear();
  };
  for (char c : over) {
    if (c == ' ' || c == '\n' || c == '\t') flush();
    else cur.push_back(c);
  }
  flush();
  std::vector<WGPUConstantEntry> constants(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    constants[i] = WGPU_CONSTANT_ENTRY_INIT;
    constants[i].key = Sv(keys[i]);
    constants[i].value = vals[i];
  }
  WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
  cd.layout = pl;
  cd.label = Sv(name);
  cd.compute.module = g.shaders[sit->second - 1];
  cd.compute.entryPoint = Sv(entry);
  cd.compute.constantCount = constants.size();
  cd.compute.constants = constants.empty() ? nullptr : constants.data();
  g.err.clear();
  WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(g.device, &cd);
  if (!pipe || !TakeErr(err)) {
    if (err->empty()) *err = "pipe: " + name + " was not created\n";
    if (pipe) wgpuComputePipelineRelease(pipe);
    wgpuPipelineLayoutRelease(pl);
    wgpuBindGroupLayoutRelease(bgl);
    return false;
  }
  Pipe p;
  p.pipe = pipe;
  p.layout = pl;
  p.bgl = bgl;
  p.name = name;
  p.keys = std::move(keys);
  p.nbind = static_cast<int>(nb);
  p.ndyn = ndyn;
  g.pipes.push_back(std::move(p));
  uint32_t id = static_cast<uint32_t>(g.pipes.size());
  g.pipeID[name] = id;
  replyID(id);
  return true;
}

static bool DoGroup(R& r, std::string* err) {
  std::string name = r.str();
  std::string pipe = r.str();
  uint32_t n = r.u32();
  struct Slot {
    std::string buf;
    uint32_t off;
    uint32_t size;
  };
  std::vector<Slot> slots(n);
  for (uint32_t i = 0; i < n; ++i) {
    slots[i].buf = r.str();
    slots[i].off = r.u32();
    slots[i].size = r.u32();
  }
  if (!r.ok) {
    *err = "group: short read\n";
    return false;
  }
  auto pit = g.pipeID.find(pipe);
  if (pit == g.pipeID.end()) {
    *err = "group: no pipe " + pipe + "\n";
    return false;
  }
  Pipe& p = g.pipes[pit->second - 1];
  if (static_cast<int>(n) != p.nbind) {
    *err = "group: " + name + " has " + std::to_string(n) + " bindings, pipe wants " + std::to_string(p.nbind) + "\n";
    return false;
  }
  std::vector<WGPUBindGroupEntry> entries(n);
  for (uint32_t i = 0; i < n; ++i) {
    auto bit = g.bufID.find(slots[i].buf);
    if (bit == g.bufID.end()) {
      *err = "group: no buffer " + slots[i].buf + "\n";
      return false;
    }
    Buf& b = g.bufs[bit->second - 1];
    uint64_t off = slots[i].off;
    uint64_t sz = slots[i].size ? slots[i].size : (b.size - off);
    if (off + sz > b.size) {
      *err = "group: " + slots[i].buf + " binding is past the buffer\n";
      return false;
    }
    WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
    e.binding = i;
    e.buffer = b.b;
    e.offset = off;
    e.size = sz;
    entries[i] = e;
  }
  WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
  bg.layout = p.bgl;
  bg.entryCount = entries.size();
  bg.entries = entries.data();
  bg.label = Sv(name);
  g.err.clear();
  WGPUBindGroup group = wgpuDeviceCreateBindGroup(g.device, &bg);
  if (!group || !TakeErr(err)) {
    if (err->empty()) *err = "group: " + name + " was not created\n";
    return false;
  }
  g.groups.push_back(Group{group, name, pit->second});
  uint32_t id = static_cast<uint32_t>(g.groups.size());
  g.groupID[name] = id;
  replyID(id);
  return true;
}

struct Op {
  uint8_t tag;
  uint32_t a, b, c, d, e;
  uint32_t dyn;
};

struct Get {
  uint32_t buf, off, n;
};

struct Wr {
  uint32_t buf, off, n;
  const uint8_t* data;
};

static bool ReadOps(R& r, std::vector<Op>& ops, std::string* err) {
  uint32_t nops = r.u32();
  if (!r.ok || nops > (1u << 24)) {
    *err = "run: op count\n";
    return false;
  }
  ops.resize(nops);
  for (uint32_t i = 0; i < nops; ++i) {
    Op& op = ops[i];
    op.tag = r.u8();
    op.dyn = 0;
    if (op.tag != 1 && op.tag != 2 && op.tag != 3) {
      *err = "run: bad op\n";
      return false;
    }
    op.a = r.u32();
    op.b = r.u32();
    op.c = r.u32();
    op.d = r.u32();
    op.e = r.u32();
    if (op.tag == 3) op.dyn = r.u32();
  }
  return r.ok;
}

static bool ReadGets(R& r, std::vector<Get>& gets) {
  uint32_t nget = r.u32();
  if (!r.ok || nget > 4096) return false;
  gets.resize(nget);
  for (uint32_t i = 0; i < nget; ++i) {
    gets[i].buf = r.u32();
    gets[i].off = r.u32();
    gets[i].n = r.u32();
  }
  return r.ok;
}

static bool EnsureStaging(uint64_t n, std::string* err) {
  if (g.staging && g.stagingSize >= n) return true;
  if (g.staging) {
    wgpuBufferDestroy(g.staging);
    wgpuBufferRelease(g.staging);
    g.staging = nullptr;
  }
  uint64_t size = 1u << 16;
  while (size < n) size <<= 1;
  WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
  bd.label = Sv("staging");
  bd.size = size;
  bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  g.staging = wgpuDeviceCreateBuffer(g.device, &bd);
  if (!g.staging) {
    *err = "run: staging buffer\n";
    return false;
  }
  g.stagingSize = size;
  return true;
}

// Exec records ops into one command buffer, copies every get into the staging
// buffer in that same command buffer, submits once, and waits once. With no gets
// and wait false it returns at once: the queue orders the next run after this one.
static bool Exec(const std::vector<Wr>& writes, const std::vector<Op>& ops, const std::vector<Get>& gets, bool wait,
                 std::string* err) {
  g.err.clear();
  for (const Wr& w : writes) {
    if (!WriteAt(w.buf, w.off, w.data, w.n, err)) return false;
  }
  uint64_t total = 0;
  std::vector<uint64_t> at(gets.size());
  for (size_t i = 0; i < gets.size(); ++i) {
    const Get& gt = gets[i];
    if (gt.buf == 0 || gt.buf > g.bufs.size()) {
      *err = "run: get buffer id\n";
      return false;
    }
    if ((gt.off & 3u) != 0 || static_cast<uint64_t>(gt.off) + RoundUp(4, gt.n) > g.bufs[gt.buf - 1].size) {
      *err = "run: get past the end of " + g.bufs[gt.buf - 1].name + "\n";
      return false;
    }
    at[i] = total;
    total += RoundUp(4, gt.n);
  }
  if (total && !EnsureStaging(total, err)) return false;
  const double t0 = g_time ? NowMs() : 0;

  WGPUCommandEncoderDescriptor ed = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g.device, &ed);
  WGPUComputePassEncoder pass = nullptr;
  auto endPass = [&]() {
    if (!pass) return;
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    pass = nullptr;
  };
  uint32_t dynOffs[4] = {0, 0, 0, 0};
  for (const Op& op : ops) {
    if (op.tag == 2) {
      endPass();
      if (op.a == 0 || op.a > g.bufs.size() || op.c == 0 || op.c > g.bufs.size()) {
        *err = "run: copy buffer id\n";
        wgpuCommandEncoderRelease(enc);
        return false;
      }
      wgpuCommandEncoderCopyBufferToBuffer(enc, g.bufs[op.a - 1].b, op.b, g.bufs[op.c - 1].b, op.d, op.e);
      continue;
    }
    if (op.a == 0 || op.a > g.pipes.size() || op.b == 0 || op.b > g.groups.size()) {
      *err = "run: dispatch id\n";
      endPass();
      wgpuCommandEncoderRelease(enc);
      return false;
    }
    const Pipe& p = g.pipes[op.a - 1];
    if (!pass) pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, p.pipe);
    if (p.ndyn > 0) {
      dynOffs[0] = op.dyn;
      wgpuComputePassEncoderSetBindGroup(pass, 0, g.groups[op.b - 1].g, static_cast<size_t>(p.ndyn), dynOffs);
    } else {
      wgpuComputePassEncoderSetBindGroup(pass, 0, g.groups[op.b - 1].g, 0, nullptr);
    }
    wgpuComputePassEncoderDispatchWorkgroups(pass, op.c ? op.c : 1, op.d ? op.d : 1, op.e ? op.e : 1);
  }
  endPass();
  for (size_t i = 0; i < gets.size(); ++i) {
    const Get& gt = gets[i];
    wgpuCommandEncoderCopyBufferToBuffer(enc, g.bufs[gt.buf - 1].b, gt.off, g.staging, at[i], RoundUp(4, gt.n));
  }
  WGPUCommandBufferDescriptor cd = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cd);
  const double t1 = g_time ? NowMs() : 0;
  wgpuQueueSubmit(g.queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);
  const double t2 = g_time ? NowMs() : 0;
  struct Timer {
    double t0, t1, t2;
    size_t n;
    ~Timer() {
      if (g_time) std::fprintf(stderr, "dawnrt run  ops %zu  encode %.2f ms  submit %.2f ms  wait %.2f ms\n", n, t1 - t0, t2 - t1, NowMs() - t2);
    }
  } timer{t0, t1, t2, ops.size()};

  if (!total) {
    if (wait) {
      struct Ctx {
        bool done = false;
      } ctx;
      WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
      cb.mode = g.timed ? WGPUCallbackMode_WaitAnyOnly : WGPUCallbackMode_AllowSpontaneous;
      cb.userdata1 = &ctx;
      cb.callback = [](WGPUQueueWorkDoneStatus, void* u1, void*) { static_cast<Ctx*>(u1)->done = true; };
      WGPUFuture f = wgpuQueueOnSubmittedWorkDone(g.queue, cb);
      Wait(f, &ctx.done);
    }
    if (!TakeErr(err)) return false;
    replyRaw(0, nullptr, 0);
    return true;
  }

  // Mapping waits for the submit that wrote the staging buffer.
  struct Map {
    bool done = false;
    bool ok = false;
    std::string msg;
  } map;
  WGPUBufferMapCallbackInfo mb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
  mb.mode = g.timed ? WGPUCallbackMode_WaitAnyOnly : WGPUCallbackMode_AllowSpontaneous;
  mb.userdata1 = &map;
  mb.callback = [](WGPUMapAsyncStatus status, WGPUStringView msg, void* u1, void*) {
    Map& m = *static_cast<Map*>(u1);
    m.ok = status == WGPUMapAsyncStatus_Success;
    m.msg = ToStr(msg);
    m.done = true;
  };
  WGPUFuture f = wgpuBufferMapAsync(g.staging, WGPUMapMode_Read, 0, static_cast<size_t>(total), mb);
  Wait(f, &map.done);
  if (!map.ok) {
    *err = "read: " + (map.msg.empty() ? std::string("map failed") : map.msg) + "\n";
    return false;
  }
  const uint8_t* p = static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(g.staging, 0, static_cast<size_t>(total)));
  if (!p) {
    *err = "read: no range\n";
    wgpuBufferUnmap(g.staging);
    return false;
  }
  if (!TakeErr(err)) {
    wgpuBufferUnmap(g.staging);
    return false;
  }
  uint64_t n = 0;
  for (const Get& gt : gets) n += gt.n;
  g_out.resize(8 + n);
  uint32_t rc = 0, n32 = static_cast<uint32_t>(n);
  std::memcpy(&g_out[0], &rc, 4);
  std::memcpy(&g_out[4], &n32, 4);
  uint64_t o = 8;
  for (size_t i = 0; i < gets.size(); ++i) {
    std::memcpy(&g_out[o], p + at[i], gets[i].n);
    o += gets[i].n;
  }
  wgpuBufferUnmap(g.staging);
  wr(g_out.data(), g_out.size());
  std::fflush(stdout);
  return true;
}

// Op 5: ops, then gets. Always waits.
static bool DoRun(R& r, std::string* err) {
  std::vector<Op> ops;
  std::vector<Get> gets;
  if (!ReadOps(r, ops, err)) {
    if (err->empty()) *err = "run: short read\n";
    return false;
  }
  if (!ReadGets(r, gets)) {
    *err = "run: short read\n";
    return false;
  }
  return Exec({}, ops, gets, true, err);
}

// Op 9: flags, writes, ops, gets. Flag 1 waits for the device even with no gets.
static bool DoRun2(R& r, std::string* err) {
  uint32_t flags = r.u32();
  uint32_t nw = r.u32();
  if (!r.ok || nw > 65536) {
    *err = "run2: short read\n";
    return false;
  }
  std::vector<Wr> writes(nw);
  for (uint32_t i = 0; i < nw; ++i) {
    writes[i].buf = r.u32();
    writes[i].off = r.u32();
    writes[i].n = r.u32();
    writes[i].data = r.bytes(writes[i].n);
  }
  std::vector<Op> ops;
  std::vector<Get> gets;
  if (!r.ok || !ReadOps(r, ops, err) || !ReadGets(r, gets)) {
    if (err->empty()) *err = "run2: short read\n";
    return false;
  }
  return Exec(writes, ops, gets, (flags & 1u) != 0, err);
}

static bool DoInfo(std::string* err) {
  (void)err;
  uint32_t v[2] = {g.info, g.uniAlign};
  replyRaw(0, reinterpret_cast<const char*>(v), sizeof(v));
  return true;
}

static bool Handle(const std::vector<uint8_t>& msg, std::string* err) {
  if (msg.size() < 4) {
    *err = "short command\n";
    return false;
  }
  R r{msg.data() + 4, msg.size() - 4};
  uint32_t op = 0;
  std::memcpy(&op, msg.data(), 4);
  switch (op) {
    case 1: return DoShader(r, err);
    case 2: return DoBuf(r, err);
    case 3: return DoPipe(r, err);
    case 4: return DoGroup(r, err);
    case 5: return DoRun(r, err);
    case 7: return DoWrite(r, err);
    case 8: return DoInfo(err);
    case 9: return DoRun2(r, err);
  }
  *err = "unknown op\n";
  return false;
}

int main(int argc, char** argv) {
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
  std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
  std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
  timeBeginPeriod(1);
  g_time = EnvOn("MODELRUN_DAWN_TIME");
  std::string err;
  if (!Open(&err)) {
    std::fprintf(stderr, "%s", err.c_str());
    return 1;
  }
  if (argc > 1 && std::strcmp(argv[1], "info") == 0) {
    std::fprintf(stderr, "dawnrt ready  info 0x%x  timed %d  uniform align %u\n", g.info, g.timed ? 1 : 0, g.uniAlign);
    return 0;
  }
  std::vector<uint8_t> msg;
  for (;;) {
    uint32_t n = 0;
    if (!rd(&n, 4)) break;
    if (n == 0) continue;
    if (n > 0x80000000u + 4096u) {
      reply(1, "command too large\n");
      return 1;
    }
    msg.resize(n);
    if (!rd(msg.data(), n)) break;
    if (n >= 4 && std::memcmp(msg.data(), "quit", 4) == 0) {
      reply(0, {});
      break;
    }
    std::string e;
    if (!Handle(msg, &e)) reply(1, e.empty() ? "failed\n" : e);
    if (msg.capacity() > (64u << 20)) std::vector<uint8_t>().swap(msg);  // an upload does not pin its size
  }
  return 0;
}
