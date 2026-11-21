package hodge

import "fmt"

// lastTokens and lastSecs are the decode that just finished.
// Prefill is not included. Stop tokens are not included.
var (
	lastTokens int
	lastSecs   float64
)

func noteSpeed(tokens int, secs float64) {
	lastTokens = tokens
	lastSecs = secs
}

// SpeedLine is the decode rate of the last reply, such as "42 tok/s".
func SpeedLine() string {
	liveMu.Lock()
	defer liveMu.Unlock()
	return formatSpeed(lastTokens, lastSecs)
}

func formatSpeed(tokens int, secs float64) string {
	if tokens <= 0 || secs <= 0 {
		return ""
	}
	rate := float64(tokens) / secs
	if rate >= 10 {
		return fmt.Sprintf("%.0f tok/s", rate)
	}
	return fmt.Sprintf("%.1f tok/s", rate)
}
