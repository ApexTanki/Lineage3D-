#include "assets/ParticleFormat.h"

#include <yyjson.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>

namespace assets {
namespace {

constexpr const char* kBlendNames[] = {"alpha", "additive", "premultiplied", "distort"};
constexpr const char* kBillboardNames[] = {"camera", "axis", "velocity", "world"};
constexpr const char* kShapeNames[] = {"point", "sphere", "disc", "box"};
constexpr float kDegrees = 3.14159265358979f / 180.0f;

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

// Messages name the emitter ("emitter \"smoke\": ...") so an author finds the line at once.
struct Reader {
    std::string* error;
    std::string where;

    bool bad(const std::string& message) const { return fail(error, where + ": " + message); }

    // Every key of `object` must be one of `allowed`, once: a typo must not silently drop a setting, and of a key given
    // twice only the first would count.
    bool onlyKnownKeys(yyjson_val* object, std::initializer_list<const char*> allowed) const {
        yyjson_obj_iter it = yyjson_obj_iter_with(object);
        size_t index = 0;
        while (yyjson_val* key = yyjson_obj_iter_next(&it)) {
            const char* k = yyjson_get_str(key);
            bool known = false;
            for (const char* a : allowed) known = known || std::strcmp(k, a) == 0;
            if (!known) return bad(std::string("unknown key \"") + k + "\"");
            yyjson_obj_iter earlier = yyjson_obj_iter_with(object);
            for (size_t i = 0; i < index; ++i)
                if (std::strcmp(yyjson_get_str(yyjson_obj_iter_next(&earlier)), k) == 0) return bad(std::string("\"") + k + "\" is given twice");
            ++index;
        }
        return true;
    }
    bool number(yyjson_val* v, const char* name, double lo, double hi, float& out) const {
        if (!v) return true;
        if (!yyjson_is_num(v)) return bad(std::string(name) + " must be a number");
        const double d = yyjson_get_num(v);
        if (!std::isfinite(d) || d < lo || d > hi) return bad(std::string(name) + " is out of range");
        out = float(d);
        return true;
    }
    bool integer(yyjson_val* v, const char* name, uint32_t lo, uint32_t hi, uint32_t& out) const {
        if (!v) return true;
        const double d = yyjson_is_num(v) ? yyjson_get_num(v) : -1.0;
        if (!std::isfinite(d) || d != std::floor(d) || d < double(lo) || d > double(hi))
            return bad(std::string(name) + " must be a whole number from " + std::to_string(lo) + " to " + std::to_string(hi));
        out = uint32_t(d);
        return true;
    }
    bool boolean(yyjson_val* v, const char* name, bool& out) const {
        if (!v) return true;
        if (!yyjson_is_bool(v)) return bad(std::string(name) + " must be true or false");
        out = yyjson_get_bool(v);
        return true;
    }
    bool floats(yyjson_val* v, const char* name, float* out, size_t count, double lo, double hi) const {
        if (!yyjson_is_arr(v) || yyjson_arr_size(v) != count)
            return bad(std::string(name) + " must be an array of " + std::to_string(count) + " numbers");
        for (size_t i = 0; i < count; ++i) {
            yyjson_val* c = yyjson_arr_get(v, i);
            const double d = yyjson_is_num(c) ? yyjson_get_num(c) : std::nan("");
            if (!std::isfinite(d) || d < lo || d > hi) return bad(std::string(name) + " is out of range");
            out[i] = float(d);
        }
        return true;
    }
    // [min, max] with min ≤ max.
    bool range(yyjson_val* v, const char* name, double lo, double hi, glm::vec2& out) const {
        if (!v) return true;
        if (!floats(v, name, &out.x, 2, lo, hi)) return false;
        if (out.x > out.y) return bad(std::string(name) + " must be [min, max] with min ≤ max");
        return true;
    }
    bool direction(yyjson_val* v, const char* name, glm::vec3& out) const {
        if (!v) return true;
        if (!floats(v, name, &out.x, 3, -1e3, 1e3)) return false;
        const float length = glm::length(out);
        if (!(length > 1e-6f)) return bad(std::string(name) + " must not be zero");
        out /= length;
        return true;
    }

    // A scalar curve: a number (constant) or keys [[t, v], ...] with t strictly increasing in [0, 1].
    bool scalarCurve(yyjson_val* v, const char* name, double lo, double hi, ScalarCurve& out) const {
        if (!v) return true;
        if (yyjson_is_num(v)) {
            float c = 0.0f;
            if (!number(v, name, lo, hi, c)) return false;
            out.samples.fill(c);
            return true;
        }
        float times[kMaxCurveKeys], values[kMaxCurveKeys];
        uint32_t count = 0;
        if (!keys(v, name, times, count)) return false;
        for (uint32_t k = 0; k < count; ++k) {
            yyjson_val* value = yyjson_arr_get(yyjson_arr_get(v, k), 1);
            const double d = yyjson_is_num(value) ? yyjson_get_num(value) : std::nan("");
            if (!std::isfinite(d) || d < lo || d > hi) return bad(std::string(name) + " has a value out of range");
            values[k] = float(d);
        }
        bake(times, values, count, out);
        return true;
    }
    // A color curve: [r, g, b, a] (constant) or keys [[t, [r, g, b, a]], ...]; channels in [0, 1].
    bool colorCurve(yyjson_val* v, const char* name, ColorCurve& out) const {
        if (!v) return true;
        if (yyjson_is_arr(v) && yyjson_arr_size(v) == 4 && yyjson_is_num(yyjson_arr_get(v, 0))) {
            glm::vec4 c;
            if (!floats(v, name, &c.x, 4, 0.0, 1.0)) return bad(std::string(name) + " channels must be from 0 to 1 (use intensity to brighten)");
            out.samples.fill(c);
            return true;
        }
        float times[kMaxCurveKeys];
        glm::vec4 values[kMaxCurveKeys];
        uint32_t count = 0;
        if (!keys(v, name, times, count)) return false;
        for (uint32_t k = 0; k < count; ++k)
            if (!floats(yyjson_arr_get(yyjson_arr_get(v, k), 1), name, &values[k].x, 4, 0.0, 1.0))
                return bad(std::string(name) + " keys must be [t, [r, g, b, a]] with channels from 0 to 1 (use intensity to brighten)");
        bake(times, values, count, out);
        return true;
    }

private:
    bool keys(yyjson_val* v, const char* name, float* times, uint32_t& count) const {
        const char* shape = " must be a constant or 1 to 8 keys [[t, value], ...] with t increasing from 0 to 1";
        if (!yyjson_is_arr(v) || yyjson_arr_size(v) < 1 || yyjson_arr_size(v) > kMaxCurveKeys) return bad(name + std::string(shape));
        count = uint32_t(yyjson_arr_size(v));
        for (uint32_t k = 0; k < count; ++k) {
            yyjson_val* key = yyjson_arr_get(v, k);
            if (!yyjson_is_arr(key) || yyjson_arr_size(key) != 2 || !yyjson_is_num(yyjson_arr_get(key, 0))) return bad(name + std::string(shape));
            const double t = yyjson_get_num(yyjson_arr_get(key, 0));
            if (!std::isfinite(t) || t < 0.0 || t > 1.0 || (k > 0 && float(t) <= times[k - 1])) return bad(name + std::string(shape));
            times[k] = float(t);
        }
        return true;
    }
    // Piecewise-linear between the keys, held before the first and after the last.
    template <typename T>
    static void bake(const float* times, const T* values, uint32_t count, ParticleCurve<T>& out) {
        for (uint32_t i = 0; i < kCurveSamples; ++i) {
            const float t = float(i) / float(kCurveSamples - 1);
            uint32_t k = 0;
            while (k < count && times[k] < t) ++k;
            if (k == 0) out.samples[i] = values[0];
            else if (k == count) out.samples[i] = values[count - 1];
            else out.samples[i] = values[k - 1] + (values[k] - values[k - 1]) * ((t - times[k - 1]) / (times[k] - times[k - 1]));
        }
    }
};

template <size_t N>
bool parseName(yyjson_val* v, const char* const (&names)[N], uint8_t& out) {
    const char* s = yyjson_get_str(v);
    for (size_t i = 0; s && i < N; ++i)
        if (std::strcmp(s, names[i]) == 0) {
            out = uint8_t(i);
            return true;
        }
    return false;
}

// Data-relative: no absolute paths, drive letters or "..", like every path a data file holds.
bool dataRelative(const std::string& p) {
    return !p.empty() && p.find("..") == std::string::npos && p.find(':') == std::string::npos && p[0] != '/' && p[0] != '\\';
}

bool parseEmitter(yyjson_val* e, uint32_t index, ParticleEmitterDesc& out, std::string* error) {
    Reader r{error, "emitter " + std::to_string(index)};
    if (!yyjson_is_obj(e)) return r.bad("must be an object");
    yyjson_val* name = yyjson_obj_get(e, "name");
    if (!yyjson_is_str(name) || !*yyjson_get_str(name) || std::strlen(yyjson_get_str(name)) > 31)
        return r.bad("needs a \"name\" of 1 to 31 characters");
    out = ParticleEmitterDesc{};
    out.name = yyjson_get_str(name);
    r.where = "emitter \"" + out.name + "\"";
    if (!r.onlyKnownKeys(e, {"name", "max", "rate", "burst", "delay", "duration", "loop", "life", "offset", "shape", "radius", "extents",
                             "direction", "cone", "speed", "inherit", "gravity", "drag", "size", "sizeRandom", "color", "alpha",
                             "intensity", "rotation", "spin", "billboard", "normal", "stretch", "blend", "texture", "flipbook",
                             "soft", "lit", "collide", "lod", "light", "strength"}))
        return false;

    if (!yyjson_obj_get(e, "max")) return r.bad("needs \"max\" (its particle budget)");
    if (!yyjson_obj_get(e, "life")) return r.bad("needs \"life\" [min, max] in seconds");
    float cone = 0.0f;
    glm::vec2 rotation(0.0f), spin(0.0f);
    if (!r.integer(yyjson_obj_get(e, "max"), "max", 1, kMaxEmitterParticles, out.max) ||
        !r.number(yyjson_obj_get(e, "rate"), "rate", 0.0, 10000.0, out.rate) ||
        !r.number(yyjson_obj_get(e, "delay"), "delay", 0.0, 60.0, out.delay) ||
        !r.number(yyjson_obj_get(e, "duration"), "duration", 0.0, 3600.0, out.duration) ||
        !r.boolean(yyjson_obj_get(e, "loop"), "loop", out.loop) ||
        !r.range(yyjson_obj_get(e, "life"), "life", 0.01, 60.0, out.life) ||
        !r.direction(yyjson_obj_get(e, "direction"), "direction", out.direction) ||
        !r.number(yyjson_obj_get(e, "cone"), "cone", 0.0, 180.0, cone) ||
        !r.range(yyjson_obj_get(e, "speed"), "speed", 0.0, 1000.0, out.speed) ||
        !r.number(yyjson_obj_get(e, "inherit"), "inherit", 0.0, 2.0, out.inherit) ||
        !r.number(yyjson_obj_get(e, "gravity"), "gravity", -100.0, 100.0, out.gravity) ||
        !r.number(yyjson_obj_get(e, "drag"), "drag", 0.0, 100.0, out.drag) ||
        !r.scalarCurve(yyjson_obj_get(e, "size"), "size", 0.0, 1000.0, out.size) ||
        !r.range(yyjson_obj_get(e, "sizeRandom"), "sizeRandom", 0.01, 100.0, out.sizeRandom) ||
        !r.colorCurve(yyjson_obj_get(e, "color"), "color", out.color) ||
        !r.number(yyjson_obj_get(e, "intensity"), "intensity", 0.0, 100.0, out.intensity) ||
        !r.range(yyjson_obj_get(e, "rotation"), "rotation", -3600.0, 3600.0, rotation) ||
        !r.range(yyjson_obj_get(e, "spin"), "spin", -3600.0, 3600.0, spin) ||
        !r.number(yyjson_obj_get(e, "soft"), "soft", 0.0, 10.0, out.soft) ||
        !r.boolean(yyjson_obj_get(e, "lit"), "lit", out.lit))
        return false;
    if (!yyjson_obj_get(e, "size")) out.size.samples.fill(1.0f);
    if (!yyjson_obj_get(e, "color")) out.color.samples.fill(glm::vec4(1.0f));
    out.cone = cone * kDegrees;
    out.rotation = rotation * kDegrees;
    out.spin = spin * kDegrees;
    if (yyjson_val* alpha = yyjson_obj_get(e, "alpha")) {
        ScalarCurve a;
        if (!r.scalarCurve(alpha, "alpha", 0.0, 1.0, a)) return false;
        for (uint32_t i = 0; i < kCurveSamples; ++i) out.color.samples[i].a *= a.samples[i];
    }

    if (yyjson_val* bursts = yyjson_obj_get(e, "burst")) {
        if (!yyjson_is_arr(bursts) || yyjson_arr_size(bursts) < 1 || yyjson_arr_size(bursts) > kMaxEmitterBursts)
            return r.bad("burst must be 1 to 8 pairs [time, count]");
        for (size_t i = 0; i < yyjson_arr_size(bursts); ++i) {
            yyjson_val* b = yyjson_arr_get(bursts, i);
            if (!yyjson_is_arr(b) || yyjson_arr_size(b) != 2) return r.bad("burst must be 1 to 8 pairs [time, count]");
            ParticleBurst& burst = out.bursts[out.burstCount++];
            if (!r.number(yyjson_arr_get(b, 0), "a burst time", 0.0, 3600.0, burst.time) ||
                !r.integer(yyjson_arr_get(b, 1), "a burst count", 1, out.max, burst.count))
                return false;
            if (out.burstCount > 1 && burst.time < out.bursts[out.burstCount - 2].time) return r.bad("burst times must not decrease");
        }
    }
    if (out.rate == 0.0f && out.burstCount == 0) return r.bad("never spawns: give it a rate or a burst");
    if (out.loop && out.duration < 0.01f) return r.bad("a looping emitter needs a duration (its period) of at least 0.01 s");
    if (out.rate > 0.0f && !out.loop && out.duration <= 0.0f) return r.bad("a rate needs a duration, or loop");
    if (out.loop)
        for (uint32_t b = 0; b < out.burstCount; ++b)
            if (out.bursts[b].time >= out.duration) return r.bad("a burst of a looping emitter must come before the end of its period");

    uint8_t value = 0;
    if (yyjson_val* shape = yyjson_obj_get(e, "shape")) {
        if (!parseName(shape, kShapeNames, value)) return r.bad("shape must be point, sphere, disc or box");
        out.shape = ParticleShape(value);
    }
    const bool round = out.shape == ParticleShape::Sphere || out.shape == ParticleShape::Disc;
    if (round != (yyjson_obj_get(e, "radius") != nullptr)) return r.bad("a radius goes with (and only with) the sphere and disc shapes");
    if ((out.shape == ParticleShape::Box) != (yyjson_obj_get(e, "extents") != nullptr))
        return r.bad("extents go with (and only with) the box shape");
    if (!r.number(yyjson_obj_get(e, "radius"), "radius", 0.001, 1000.0, out.radius)) return false;
    if (yyjson_val* extents = yyjson_obj_get(e, "extents"))
        if (!r.floats(extents, "extents", &out.extents.x, 3, 0.0, 1000.0)) return false;
    if (yyjson_val* offset = yyjson_obj_get(e, "offset"))
        if (!r.floats(offset, "offset", &out.offset.x, 3, -1000.0, 1000.0)) return false;

    if (yyjson_val* billboard = yyjson_obj_get(e, "billboard")) {
        if (!parseName(billboard, kBillboardNames, value)) return r.bad("billboard must be camera, axis, velocity or world");
        out.billboard = ParticleBillboard(value);
    }
    if (yyjson_obj_get(e, "normal") && out.billboard != ParticleBillboard::World) return r.bad("a normal goes with world billboards only");
    if (yyjson_obj_get(e, "stretch") && out.billboard != ParticleBillboard::Velocity)
        return r.bad("a stretch goes with velocity billboards only");
    if (!r.direction(yyjson_obj_get(e, "normal"), "normal", out.normal) ||
        !r.number(yyjson_obj_get(e, "stretch"), "stretch", 0.0, 10.0, out.stretch))
        return false;
    if (yyjson_val* blend = yyjson_obj_get(e, "blend")) {
        if (!parseName(blend, kBlendNames, value)) return r.bad("blend must be alpha, additive, premultiplied or distort");
        out.blend = ParticleBlend(value);
    }
    if (yyjson_val* texture = yyjson_obj_get(e, "texture")) {
        const char* path = yyjson_get_str(texture);
        if (!path || !dataRelative(path)) return r.bad("texture must be a data-relative path");
        out.texture = path;
        if (out.texture.size() < 5 || out.texture.compare(out.texture.size() - 5, 5, ".ktx2") != 0) return r.bad("texture must be a .ktx2 file");
    }

    if (yyjson_val* flipbook = yyjson_obj_get(e, "flipbook")) {
        Reader f{error, r.where + " flipbook"};
        if (!yyjson_is_obj(flipbook)) return f.bad("must be an object");
        if (!f.onlyKnownKeys(flipbook, {"cols", "rows", "frames", "fps", "overLife", "loop", "interpolate", "randomStart"})) return false;
        if (!f.integer(yyjson_obj_get(flipbook, "cols"), "cols", 1, kMaxFlipbookCells, out.columns) ||
            !f.integer(yyjson_obj_get(flipbook, "rows"), "rows", 1, kMaxFlipbookCells, out.rows))
            return false;
        out.frameCount = out.columns * out.rows;
        if (!f.integer(yyjson_obj_get(flipbook, "frames"), "frames", 1, out.columns * out.rows, out.frameCount) ||
            !f.number(yyjson_obj_get(flipbook, "fps"), "fps", 0.0, 240.0, out.fps) ||
            !f.boolean(yyjson_obj_get(flipbook, "overLife"), "overLife", out.overLife) ||
            !f.boolean(yyjson_obj_get(flipbook, "loop"), "loop", out.flipbookLoop) ||
            !f.boolean(yyjson_obj_get(flipbook, "interpolate"), "interpolate", out.interpolate) ||
            !f.boolean(yyjson_obj_get(flipbook, "randomStart"), "randomStart", out.randomStart))
            return false;
        if (out.texture.empty()) return f.bad("needs the emitter's texture");
        if (out.overLife && out.fps > 0.0f) return f.bad("plays at fps or over life, not both");
        const bool still = !out.overLife && out.fps <= 0.0f; // each particle keeps one frame
        if (out.frameCount > 1 && still && !out.randomStart)
            return f.bad("needs fps or overLife to animate its frames, or randomStart for a random still frame");
        if (still && out.interpolate) return f.bad("a still frame doesn't interpolate");
        if (out.overLife && yyjson_obj_get(flipbook, "loop")) return f.bad("loop applies to fps playback only");
    }

    if (yyjson_val* collide = yyjson_obj_get(e, "collide")) {
        Reader c{error, r.where + " collide"};
        if (!yyjson_is_obj(collide)) return c.bad("must be an object");
        if (!c.onlyKnownKeys(collide, {"restitution", "friction"}) ||
            !c.number(yyjson_obj_get(collide, "restitution"), "restitution", 0.0, 1.0, out.restitution) ||
            !c.number(yyjson_obj_get(collide, "friction"), "friction", 0.0, 1.0, out.friction))
            return false;
        out.collide = true;
    }
    if (yyjson_val* lod = yyjson_obj_get(e, "lod")) {
        if (!r.floats(lod, "lod", &out.lod.x, 2, 0.0, 100000.0)) return false;
        if (!(out.lod.x < out.lod.y)) return r.bad("lod must be [near, far] with near < far");
    }
    if (yyjson_val* light = yyjson_obj_get(e, "light")) {
        Reader l{error, r.where + " light"};
        if (!yyjson_is_obj(light)) return l.bad("must be an object");
        if (!l.onlyKnownKeys(light, {"color", "intensity", "range", "life"})) return false;
        if (!yyjson_obj_get(light, "intensity") || !yyjson_obj_get(light, "range") || !yyjson_obj_get(light, "life"))
            return l.bad("needs intensity, range and life");
        if (yyjson_val* color = yyjson_obj_get(light, "color"))
            if (!l.floats(color, "color", &out.light.color.x, 3, 0.0, 1000.0)) return false;
        if (!l.number(yyjson_obj_get(light, "intensity"), "intensity", 0.0, 1000.0, out.light.intensity) ||
            !l.number(yyjson_obj_get(light, "range"), "range", 0.1, 100.0, out.light.range) ||
            !l.number(yyjson_obj_get(light, "life"), "life", 0.01, 10.0, out.light.life))
            return false;
        out.hasLight = true;
    }
    // Heat haze (M8 plan D11): a normal map bends the picture; nothing to light and nothing that lights.
    if (yyjson_val* strength = yyjson_obj_get(e, "strength")) {
        if (out.blend != ParticleBlend::Distort) return r.bad("strength goes with distort emitters only");
        if (!r.number(strength, "strength", 0.0, 2.0, out.strength)) return false;
    }
    if (out.blend == ParticleBlend::Distort) {
        if (out.texture.empty()) return r.bad("a distort emitter needs a texture (a normal map)");
        if (out.lit) return r.bad("a distort emitter can't be lit");
        if (out.hasLight) return r.bad("a distort emitter can't carry a light");
        if (out.columns * out.rows > 1) return r.bad("a distort emitter can't use a flipbook (its soft mask spans the whole quad)");
    }
    return true;
}

} // namespace

const char* particleBlendName(ParticleBlend blend) {
    return kBlendNames[uint32_t(blend)];
}

const char* particleBillboardName(ParticleBillboard billboard) {
    return kBillboardNames[uint32_t(billboard)];
}

bool parseParticleEffect(std::span<const std::byte> json, ParticleEffectFile& out, std::string* error) {
    yyjson_doc* doc = yyjson_read(reinterpret_cast<const char*>(json.data()), json.size(), 0);
    if (!doc) return fail(error, "not valid JSON");
    struct Guard {
        yyjson_doc* d;
        ~Guard() { yyjson_doc_free(d); }
    } guard{doc};
    yyjson_val* root = yyjson_doc_get_root(doc);
    const Reader r{error, "the effect"};
    if (!yyjson_is_obj(root)) return r.bad("the root must be an object");
    if (!r.onlyKnownKeys(root, {"name", "emitters"})) return false;
    out = ParticleEffectFile{};
    if (yyjson_val* name = yyjson_obj_get(root, "name")) {
        if (!yyjson_is_str(name)) return r.bad("name must be a string");
        out.name = yyjson_get_str(name);
    }
    yyjson_val* emitters = yyjson_obj_get(root, "emitters");
    if (!yyjson_is_arr(emitters) || yyjson_arr_size(emitters) < 1 || yyjson_arr_size(emitters) > kMaxEffectEmitters)
        return r.bad("emitters must be an array of 1 to 8 emitters");
    out.emitters.resize(yyjson_arr_size(emitters));
    for (size_t i = 0; i < out.emitters.size(); ++i) {
        if (!parseEmitter(yyjson_arr_get(emitters, i), uint32_t(i), out.emitters[i], error)) return false;
        for (size_t j = 0; j < i; ++j)
            if (out.emitters[j].name == out.emitters[i].name) return r.bad("two emitters are named \"" + out.emitters[i].name + "\"");
        out.loops = out.loops || out.emitters[i].loop;
        out.particleCapacity += out.emitters[i].max;
    }
    return validateParticleEffect(out, error); // a safety net: the checks above already guarantee it
}

bool validateParticleEffect(const ParticleEffectFile& effect, std::string* error) {
    if (effect.emitters.empty() || effect.emitters.size() > kMaxEffectEmitters) return fail(error, "an effect needs 1 to 8 emitters");
    auto finite = [](std::initializer_list<float> values) {
        for (float v : values)
            if (!std::isfinite(v)) return false;
        return true;
    };
    auto unit = [](const glm::vec3& v) { return std::isfinite(v.x + v.y + v.z) && std::fabs(glm::length(v) - 1.0f) < 1e-3f; };
    for (size_t i = 0; i < effect.emitters.size(); ++i) {
        const ParticleEmitterDesc& e = effect.emitters[i];
        const Reader r{error, "emitter " + std::to_string(i)};
        if (e.max < 1 || e.max > kMaxEmitterParticles) return r.bad("max must be 1 to 4096");
        if (e.burstCount > kMaxEmitterBursts) return r.bad("at most 8 bursts");
        for (uint32_t b = 0; b < e.burstCount; ++b)
            if (!(e.bursts[b].time >= 0.0f && e.bursts[b].time <= 3600.0f) || e.bursts[b].count < 1 || e.bursts[b].count > e.max)
                return r.bad("a burst needs a time from 0 to 3600 s and a count from 1 to max");
        if (!(e.rate >= 0.0f && e.rate <= 10000.0f)) return r.bad("rate must be 0 to 10000");
        if (!(e.delay >= 0.0f && e.delay <= 60.0f) || !(e.duration >= 0.0f && e.duration <= 3600.0f)) return r.bad("delay or duration out of range");
        if (e.loop && !(e.duration >= 0.01f)) return r.bad("a looping emitter needs a duration of at least 0.01 s");
        if (!(e.life.x >= 0.01f && e.life.x <= e.life.y && e.life.y <= 60.0f)) return r.bad("life must be [min, max] within 0.01–60 s");
        if (!finite({e.offset.x, e.offset.y, e.offset.z, e.radius, e.extents.x, e.extents.y, e.extents.z, e.cone, e.speed.x, e.speed.y,
                     e.inherit, e.gravity, e.drag, e.sizeRandom.x, e.sizeRandom.y, e.intensity, e.rotation.x, e.rotation.y, e.spin.x,
                     e.spin.y, e.stretch, e.fps, e.soft, e.lod.x, e.lod.y, e.restitution, e.friction}))
            return r.bad("non-finite value");
        if (e.radius < 0.0f || e.extents.x < 0.0f || e.extents.y < 0.0f || e.extents.z < 0.0f || e.cone < 0.0f || e.cone > 3.1416f ||
            e.speed.x < 0.0f || e.speed.x > e.speed.y || e.drag < 0.0f || !(e.sizeRandom.x > 0.0f) || e.sizeRandom.x > e.sizeRandom.y ||
            e.intensity < 0.0f || e.stretch < 0.0f || e.fps < 0.0f || e.soft < 0.0f || e.restitution < 0.0f || e.restitution > 1.0f ||
            e.friction < 0.0f || e.friction > 1.0f || e.lod.x < 0.0f || (e.lod.y > 0.0f && !(e.lod.x < e.lod.y)))
            return r.bad("value out of range");
        if (!unit(e.direction) || !unit(e.normal)) return r.bad("direction and normal must be unit vectors");
        if (e.columns < 1 || e.columns > kMaxFlipbookCells || e.rows < 1 || e.rows > kMaxFlipbookCells || e.frameCount < 1 ||
            e.frameCount > e.columns * e.rows)
            return r.bad("the flipbook must fit its grid (1–32 columns and rows)");
        for (uint32_t s = 0; s < kCurveSamples; ++s) {
            const glm::vec4& c = e.color.samples[s];
            if (!(e.size.samples[s] >= 0.0f && e.size.samples[s] <= 1000.0f)) return r.bad("size samples must be 0 to 1000 m");
            if (!(c.r >= 0.0f && c.r <= 1.0f && c.g >= 0.0f && c.g <= 1.0f && c.b >= 0.0f && c.b <= 1.0f && c.a >= 0.0f && c.a <= 1.0f))
                return r.bad("color samples must be 0 to 1");
        }
        if (e.hasLight && (!finite({e.light.color.r, e.light.color.g, e.light.color.b, e.light.intensity}) || e.light.color.r < 0.0f ||
                           e.light.color.g < 0.0f || e.light.color.b < 0.0f || e.light.intensity < 0.0f ||
                           !(e.light.range >= 0.1f && e.light.range <= 100.0f) || !(e.light.life >= 0.01f && e.light.life <= 10.0f)))
            return r.bad("the light's values are out of range");
        // Heat haze (M8 plan D11), as the parser has it.
        if (!(e.strength >= 0.0f && e.strength <= 2.0f)) return r.bad("strength must be 0 to 2 m");
        if (e.blend == ParticleBlend::Distort && (e.texture.empty() || e.lit || e.hasLight || e.columns * e.rows > 1))
            return r.bad("a distort emitter needs a texture without a flipbook, isn't lit and carries no light");
    }
    return true;
}

} // namespace assets
