#pragma once

// Tracy CPU zones. They compile to nothing when FL_TRACY is off (Release builds).
#if FL_TRACY
#include <tracy/Tracy.hpp>
#define FL_ZONE(name) ZoneScopedN(name)
#define FL_FRAME_MARK() FrameMark
#else
#define FL_ZONE(name)
#define FL_FRAME_MARK()
#endif
