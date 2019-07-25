#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "helpers.h"
#include "spline.h"

using std::map;
using std::tuple;
using std::vector;

// Map waypoints in x, y, s, dx and dy.
struct MapData {
  vector<double> xs;
  vector<double> ys;
  vector<double> ss;
  vector<double> dxs;
  vector<double> dys;
};

// Vehicle state.
struct Pose {
  double x;      // m
  double y;      // m
  double yaw;    // rad
  double s;
  double d;
  double speed;  // m/s
};

// Point on x-y plane.
struct Point {
  double x;
  double y;
};

// Gets distance between points |p1| and |p2|.
double Distance(const Point& p1, const Point& p2) {
  return distance(p1.x, p1.y, p2.x, p2.y);
}

// Get angle in rad from point |p1| to |p2|. 
double Angle(const Point& p1, const Point& p2) {
  return atan2(p2.y - p1.y, p2.x - p1.x);
}

// Transforms point |p| from world coordinate into |ref_pose| coordinate.
Point ToRefCoord(const Point& p, const Pose& ref_pose) {
  const double yaw = ref_pose.yaw;
  const double shift_x = p.x - ref_pose.x;
  const double shift_y = p.y - ref_pose.y;
  return {
    shift_x*cos(-yaw) - shift_y*sin(-yaw),  // x
    shift_x*sin(-yaw) + shift_y*cos(-yaw)   // y
  };
}

// Transforms point |p| from |ref_pose| coordinate back to world coordinate.
Point FromRefCoord(const Point& p, const Pose& ref_pose) {
  const double yaw = ref_pose.yaw;
  const double rot_x = p.x*cos(yaw) - p.y*sin(yaw);
  const double rot_y = p.x*sin(yaw) + p.y*cos(yaw);
  return {
    rot_x + ref_pose.x,  // x
    rot_y + ref_pose.y   // y
  };
}

using Polyline = vector<Point>;

// Converts vectors of x-values and y-values in to a Polyline.
Polyline XYVectorsToPolyline(const vector<double>& xs, const vector<double>& ys) {
  Polyline line;
  for (int i=0; i<xs.size(); ++i) {
    line.push_back({xs[i], ys[i]});
  }
  return line;
}

// Converts a Polyline into vectors of x-values and y-values.
std::pair<vector<double>, vector<double>> PolylineToXYVectors(const Polyline& line) {
  vector<double> xs;
  vector<double> ys;
  for (const auto& point : line) {
    xs.push_back(point.x);
    ys.push_back(point.y);
  }
  return std::make_pair(xs, ys);
}

// Class to generate trajectory.
class Planner {

// Planner state.
enum class State {
  KEEP_LANE,
  LANE_CHANGING
}; 

public:
  Planner(const MapData& map_data, double max_s, int track_lane = 1)
      : map_data_(map_data), MAX_S(max_s), track_lane_(track_lane), state_(State::KEEP_LANE) {}

  Polyline Plan(const Polyline& input_prev_path, const Pose &ego_car, const std::map<int, Pose>& vehicles);

private:
  // Normalize Frenet s-value in range [0, MAX_S].
  double NormalizeS(double s) {
    while (s > MAX_S) { s -= MAX_S; }
    while (s < 0) { s += MAX_S; }
    return s;
  }

  // Returns true if |s1| is further than |s2| (Frenet s-value).
  // Lap wrap is considered.
  bool IsFurtherInS(double s1, double s2) {
    s1 = NormalizeS(s1);
    s2 = NormalizeS(s2);
    if (std::abs(s1 - s2) > MAX_S * 0.5) {
      if (s1 < s2) {
        s1 += MAX_S;
      } else if (s1 > s2) {
        s2 += MAX_S;
      }
    }
    return s1 > s2;
  }
  
  // Returns the distance between two Frenet s-values.
  // Lap wrap is considered.
  double DistanceInS(double s1, double s2) {
    s1 = NormalizeS(s1);
    s2 = NormalizeS(s2);
    double dist = std::abs(s1 - s2);
    if (dist > MAX_S * 0.5) {
      dist = MAX_S - dist;
    }
    return dist;
  }

  // Cvonerts |lane| (index) to Frenet d-value.
  double LaneToD(int lane) {
    return lane * 4.0 + 2.0;
  }

  bool InLane(const Pose& ego_car, int lane) {
    double target_d = LaneToD(lane);
    return std::abs(ego_car.d - target_d) < 0.5; 
  }

  // Converts Frenet to Cartesian Point.
  Point FrenetToXY(double s, double d) {
    s = NormalizeS(s);
    auto waypt = getXY(s, d, map_data_.ss, map_data_.xs, map_data_.ys);
    return {waypt[0], waypt[1]};
  }

  // Converts Cartesian |x|, |y| and |theta| to Frenet.
  vector<double> XYToFrenet(double x, double y, double theta) {
    return getFrenet(x, y, theta, map_data_.xs, map_data_.ys);
  }

  // Creates a feasible trajectory with target lane and speed.
  Polyline CreateTrajectory(
    const Polyline& prev_path, const Pose& ego_car, 
    int target_lane, double target_speed);

  // Gets maximum feasible speed at |lane| without colliding vehicle in front. 
  double GetTargetSpeed(int lane, const Pose& ego_car, const std::map<int, Pose>& vehicles);

  // Checks whether there is any vehicle in given lane between |s_range|.
  // |s_range| is a pair of (s_begin, s_end).
  bool HasVehicle(int lane, std::pair<double, double> s_range, const std::map<int, Pose>& vehicles);

  // Cost function of a lane - trajectory. The smaller the better.
  double GetCost(int lane, const Polyline& trajectory, const Pose& ego_car);

  MapData map_data_;

  State state_;

  // The lane ego car trying to track.
  int track_lane_;

  // Constants used in planner.
  // The max s value before wrapping around the track back to 0.
  double MAX_S = 6945.554;

  double MAX_SPEED = 21.5;    // 22 m/s = 49.21 mph
  double MIN_SPEED = 20.5;    
  double MAX_ACCEL = 10;    // 10 m/s/s
  double MAX_JERK = 10;     // 10 m/s/s/s

  // Delta time between each point in trajectory.
  double DELTA_T = 0.02;  // second
  size_t TRAJECTORY_SIZE = 50;    // 50 * 0.02 ms = 1 sec
  // Number of points stiched from previous path.
  size_t STITCH_SIZE = 50;

  size_t NUM_LANE = 3;

  // Vehicle dimension.
  double VEHICLE_WIDTH = 0.75; 
  
  // Number of steps extended for refernce line and step distance (in s-value).
  size_t NUM_REF_STEP = 3;
  double REF_STEP_S = 30.0; 

  // Safety fence to front, side front and side rear vehicles (in s-value).
  double FRONT_FENCE_S = MAX_SPEED * 2.0;
  double SIDE_FRONT_FENCE_S = 12.0;
  double SIDE_REAR_FENCE = 8.0;
};

Polyline Planner::Plan(
    const Polyline& input_prev_path, 
    const Pose& ego_car,
    const std::map<int, Pose>& vehicles) {
  
  // Truncate prev_path to STITCH_SIZE.
  Polyline prev_path(input_prev_path.begin(), 
                     input_prev_path.begin() + std::min(input_prev_path.size(), STITCH_SIZE));
  
  // Generate trajectory on |track_lane_|.
  const double target_speed = GetTargetSpeed(track_lane_, ego_car, vehicles);
  // std::cout << "target speed: " << target_speed << std::endl;
  Polyline trajectory = CreateTrajectory(prev_path, ego_car, track_lane_, target_speed);
  

  std::cout << "target speed: " << target_speed << std::endl;



  if (state_ == State::LANE_CHANGING) {
    // If LANE_CHANGING, just finish ongoing lane changing.
    if (InLane(ego_car, track_lane_)) {
      // Transit to KEEP_LANE once ego car d-value is closest to lane center enough. 
      state_ = State::KEEP_LANE;
    }
    return trajectory;
  }
  
  // Return the trajectory is speed is acceptable.
  if (target_speed >= MIN_SPEED) {
    return trajectory;
  }

  // If speed is too low, try to find better candidate trajectoies.
  map<int, Polyline> lane_trajectories;
  lane_trajectories.emplace(track_lane_, std::move(trajectory));

  // TODO: tuning, or use vehicle's speed
  std::pair<double, double> safe_s_range = {
    ego_car.s - SIDE_FRONT_FENCE_S,
    ego_car.s + SIDE_REAR_FENCE
  };

  for (int lane_change = -1; lane_change <= 1; lane_change+=2) {
    const int cand_lane = track_lane_ + lane_change;
    if (cand_lane < 0 || cand_lane >= NUM_LANE) {
      continue;
    }

    if (HasVehicle(cand_lane, safe_s_range, vehicles)) {
      // Too danger to make lange chagne, skipped.
      continue;
    }

    const double cand_speed = GetTargetSpeed(cand_lane, ego_car, vehicles);      
    Polyline cand_trajectory = CreateTrajectory(prev_path, ego_car, cand_lane, cand_speed);
    lane_trajectories.emplace(cand_lane, std::move(cand_trajectory));
  }

  // Pick the best trajectory with lowest cost.
  int best_lane;
  double best_cost = std::numeric_limits<double>::max(); 
  for (const auto& entry : lane_trajectories) {
    const double lane = entry.first;
    const Polyline& trajectory = entry.second;
    const double cost = GetTrajectoryCost(lane, trajectory, ego_car);
    if (cost < best_cost) {
      best_cost = cost;
      best_lane = lane;
    }
  }

  if (best_lane != track_lane_) {
    // Change lane.
    state_ = State::LANE_CHANGING;
    track_lane_ = best_lane;
  }
  return lane_trajectories[best_lane];
}

 
Polyline Planner::CreateTrajectory(
    const Polyline& prev_path, const Pose& ego_car, 
    int target_lane, double target_speed) {
  // Pose at the end of |prev_path|.
  Pose end_pose;
  Polyline ref_line;

  // Initialize reference line ahd refernce pose (pose at previous path last point) 
  // from the previous path's last two waypoints. 
  const int prev_size = prev_path.size();
  if (prev_size < 2) {
    end_pose = ego_car;
    ref_line.push_back({
      end_pose.x - cos(end_pose.yaw),  // x
      end_pose.y - sin(end_pose.yaw)   // y
    });
  } else {
    const Point& last_point = prev_path[prev_size - 1];
    const Point& last2_point = prev_path[prev_size - 2];

    ref_line.push_back(last2_point);

    const double yaw = Angle(last2_point, last_point);
    const auto end_frenet = XYToFrenet(last_point.x, last_point.y, yaw);

    end_pose = {
      last_point.x,
      last_point.y,
      yaw,  // yaw
      end_frenet[0],  // s
      end_frenet[1],  // d
      Distance(last_point, last2_point) / DELTA_T  // speed
    };
  }
  ref_line.push_back({end_pose.x, end_pose.y});

  // Append more points along s to refernce line.
  const double target_d = LaneToD(target_lane); 
  for (int i=1; i<=NUM_REF_STEP; ++i) {
    const double target_s = end_pose.s + i * REF_STEP_S;
    const Point waypoint = FrenetToXY(target_s, target_d);
    ref_line.push_back(waypoint);
  }

  // Transform reference line points to coordinates orginated at |end_pose|.
  for (int i = 0; i < ref_line.size(); ++i) {
    ref_line[i] = ToRefCoord(ref_line[i], end_pose);
  }

  // Fit refernce line to a c-spline.
  tk::spline spline;
  vector<double> ref_xs;
  vector<double> ref_ys;
  std::tie(ref_xs, ref_ys) = PolylineToXYVectors(ref_line);
  spline.set_points(ref_xs, ref_ys);

  // Max x distance traveled in next second.
  double horizon_x = MAX_SPEED * 1.0;
  double horizon_y = spline(horizon_x);
  double horizon_angle = atan2(horizon_y, horizon_x);

  // Initialize trajectolry by stiching points from previous path.
  Polyline trajectory(prev_path.begin(), prev_path.end());
  
  // Add waypoints to trajectory without violating speed and acceleration limit.
  double speed = end_pose.speed;
  Point spline_point = {0.0, 0.0};
  while (trajectory.size() < TRAJECTORY_SIZE) {
    spline_point.x += DELTA_T * speed * cos(horizon_angle);
    spline_point.y = spline(spline_point.x);

    Point traj_point = FromRefCoord(spline_point, end_pose);
    trajectory.push_back(traj_point);

    // Use 70% of acceleration.
    if (speed < target_speed) {
      speed += std::min(target_speed - speed, MAX_ACCEL * DELTA_T * 0.7);
    } else if (speed > target_speed) {
      speed -= std::min(speed - target_speed, MAX_ACCEL * DELTA_T * 0.7);  
    }
  }
  return trajectory;
}

double Planner::GetTargetSpeed(
    int lane, const Pose& ego_car, const std::map<int, Pose>& vehicles) {
  const double target_d = LaneToD(lane);
  
  double target_speed = MAX_SPEED;
  double min_s_dist= std::numeric_limits<double>::max();

  for (const auto& entry : vehicles) {
    const auto& vehicle = entry.second;
    const double s_dist = DistanceInS(vehicle.s, ego_car.s);
    if (abs(vehicle.d - target_d) <= VEHICLE_WIDTH &&
        IsFurtherInS(vehicle.s, ego_car.s) &&
        s_dist < FRONT_FENCE_S) {
      if (s_dist < min_s_dist) {
        min_s_dist = s_dist;
        target_speed = std::min(target_speed, vehicle.speed);
      }
    }
  }
  return target_speed;

  // double closest_s_diff = std::numeric_limits<double>::max();
  // double target_speed = MAX_SPEED;
  // double target_d = LaneToD(lane);

  // for (const auto& entry : vehicles) {
  //   const auto& vehicle = entry.second;
  //   double s_diff = vehicle.s - ego_car.s;
  //   // TODO: s range
  //   if (abs(vehicle.d - target_d) <= VEHICLE_WIDTH &&
  //       0 <= s_diff && s_diff <= FRONT_FENCE_S) {
  //     if (s_diff < closest_s_diff) {
  //       closest_s_diff = s_diff;
  //       target_speed = std::min(target_speed, vehicle.speed);
  //     }
  //   }
  // }
  // return target_speed;
}
 
bool Planner::HasVehicle(
    int lane, std::pair<double, double> s_range, const std::map<int, Pose>& vehicles) {
  const double target_d = LaneToD(lane);

  for (const auto& entry : vehicles) {
    const auto& vehicle = entry.second;
    if (abs(vehicle.d - target_d) <= VEHICLE_WIDTH &&
        IsFurtherInS(s_range.first, vehicle.s) &&
        IsFurtherInS(vehicle.s, s_range.second)) {
      return true;
    }
  }
  return false;

  // for (const auto& entry : vehicles) {
  //   const auto& vehicle = entry.second;
  //   // TODO: s range
  //   if (abs(vehicle.d - target_d) <= VEHICLE_WIDTH &&
  //       s_range.first <= vehicle.s && vehicle.s < s_range.second) {
  //     return true;
  //   }
  // }
  // return false;
}

double Planner::GetCost(int lane, const Polyline& trajectory, const Pose& ego_car) {
  const int traj_size = trajectory.size();
  
  // Get speed at the end of trajectory.
  const Point& last_point = trajectory[traj_size - 1];
  const Point& last2_point = trajectory[traj_size - 2];
  const double speed = Distance(last_point, last2_point) / DELTA_T;
  const double speed_cost = MAX_SPEED - speed;
  
  // Get total Frenet s-value has been traveled in trajectory.
  const auto end_frenet = XYToFrenet(last_point.x, last_point.y, Angle(last2_point, last_point));
  const double end_s = end_frenet[0];
  // Compensate s drecrease due to crossing the lap end.
  const double s_traveled = DistanceInS(end_s, ego_car.s);

  // if (s_traveled < 0) {
  //   s_traveled += MAX_S;
  // }
  const double dist_cost = MAX_SPEED * 2.0 - s_traveled;

  // TODO: this is shit
  // return abs(MAX_SPEED - speed) * 100 + (s_diff);

  // Honor center lane (most flexible in lane changing).
  const double lane_cost = std::abs(1 - lane) * 10;

  return speed_cost + dist_cost + lane_cost;
}