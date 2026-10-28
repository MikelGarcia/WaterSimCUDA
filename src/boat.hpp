#pragma once

namespace water {
struct BoatControl { float throttle = 0, rudder = 0; };
struct BoatState { float x, z, yaw, vx, vz, yaw_rate; };
struct BoatGeometry { float length, beam, spawn_clearance; };
inline constexpr BoatGeometry boat_geometry{0.30f, 0.113685692f, 0.0113685692f};
struct BoatPoint { float x, z; };
// Evaluated hull intersection at source Z=-0.5, simplified inward to 16 vertices.
inline constexpr BoatPoint boat_waterline[16]{
    {-0.147245352f, -0.003249567f},
    {-0.144698931f, -0.032702610f},
    {-0.139527242f, -0.049925601f},
    {-0.088544768f, -0.053872564f},
    {0.035774631f, -0.049859911f},
    {0.096570376f, -0.035659779f},
    {0.114798997f, -0.024418568f},
    {0.129889143f, -0.002314593f},
    {0.129889143f, 0.002314594f},
    {0.114798989f, 0.024418572f},
    {0.096570372f, 0.035659775f},
    {0.080346428f, 0.041421209f},
    {0.035774631f, 0.049859911f},
    {-0.088544751f, 0.053872568f},
    {-0.139527225f, 0.049925605f},
    {-0.144698948f, 0.032702612f},
};
inline constexpr int boat_waterline_count = 16;
} // namespace water
