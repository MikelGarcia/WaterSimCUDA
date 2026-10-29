#pragma once

namespace water {
struct BoatControl { float throttle = 0, rudder = 0; };
struct BoatState { float x, z, yaw, vx, vz, yaw_rate; float wake_phase = 0; };
struct BoatGeometry { float length, beam, spawn_clearance; };
inline constexpr BoatGeometry boat_geometry{0.60f, 0.227371384f, 0.0227371384f};
struct BoatPoint { float x, z; };
// Evaluated hull intersection at source Z=-0.5, simplified inward to 16 vertices.
inline constexpr BoatPoint boat_waterline[16]{
    {-0.294490704f, -0.006499134f},
    {-0.289397862f, -0.065405220f},
    {-0.279054484f, -0.099851202f},
    {-0.177089536f, -0.107745128f},
    {0.071549262f, -0.099719822f},
    {0.193140752f, -0.071319558f},
    {0.229597994f, -0.048837136f},
    {0.259778286f, -0.004629186f},
    {0.259778286f, 0.004629188f},
    {0.229597978f, 0.048837144f},
    {0.193140744f, 0.071319550f},
    {0.160692856f, 0.082842418f},
    {0.071549262f, 0.099719822f},
    {-0.177089502f, 0.107745136f},
    {-0.279054450f, 0.099851210f},
    {-0.289397896f, 0.065405224f},
};
inline constexpr int boat_waterline_count = 16;
} // namespace water
