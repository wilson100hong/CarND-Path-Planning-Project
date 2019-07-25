#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "planner.h"

// For convenience
using nlohmann::json;
using std::string;
using std::vector;
using std::map;

#define MPH_TO_MS 0.44704

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }
  
  MapData map_data = {
    map_waypoints_x,
    map_waypoints_y,
    map_waypoints_s,
    map_waypoints_dx,
    map_waypoints_dy
  };

  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  Planner planner(map_data, max_s);

  h.onMessage([&planner]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Ego car's localization Data
          Pose ego_car = {
              j[1]["x"],
              j[1]["y"],
              deg2rad(j[1]["yaw"]),  // rad
              j[1]["s"],
              j[1]["d"],
              j[1]["speed"]   // mph
          };

          ego_car.speed *= MPH_TO_MS; // m/s

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];
        
          // Sensor Fusion Data, a list of all other cars on the same side of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          // Convert sensor fusion to vehicle agent poses.
          map<int, Pose> vehicles;
          for (const auto& sensed : sensor_fusion) {
            double vx = sensed[3];
            double vy = sensed[4];
            vehicles[sensed[0]] = {  // id
              sensed[1],          // x
              sensed[2],          // y
              atan2(vy, vx),      // yaw, rad
              sensed[5],          // s
              sensed[6],          // d
              std::hypot(vx, vy)  // speed, m/s
            };
          }

          const Polyline prev_path = XYVectorsToPolyline(previous_path_x, previous_path_y);
          const Polyline trajectory = planner.Plan(prev_path, ego_car, vehicles);

          vector<double> next_xs, next_ys;
          std::tie(next_xs, next_ys) = PolylineToXYVectors(trajectory);
          
          json msgJson;
          msgJson["next_x"] = next_xs;
          msgJson["next_y"] = next_ys;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}