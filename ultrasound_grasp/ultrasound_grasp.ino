#include <FastLED.h>
#include <Servo.h>
#include <Wire.h>
#include "Ultrasound.h"
#include "tone.h"

// Hardware Pins
const uint8_t servoPins[6] = { 7, 6, 5, 4, 3, 2 };
const uint8_t rgbPin = 13;
const uint8_t buzzerPin = 11;

// Robot State & Movement
static float servo_angles[6] = { 90, 90, 90, 90, 90, 90 }; 
static uint8_t target_angles[6] = { 90, 90, 90, 90, 90, 90 };
const uint8_t limt_angles[6][2] = {{0,90},{0,180},{0,180},{25,180},{0,180},{0,180}}; 

// Objects
Servo servos[6];
Ultrasound ultrasound;
CRGB rgbs[1];

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);

  // Initialize Servos
  for (int i = 0; i < 6; ++i) {
    servos[i].attach(servoPins[i]);
  }

  // Initialize LEDs
  FastLED.addLeds<WS2812, rgbPin, GRB>(rgbs, 1);
  rgbs[0] = CRGB::Blue;
  FastLED.show();

  // Initial Pose: Open Hand / Ready
  set_pose(90, 90, 0, 90, 90, 90); 
  
  // Audio Feedback
  tone(buzzerPin, 1000); 
  delay(100);
  noTone(buzzerPin); 

  Serial.println("Ultrasound Grasp Demo Starting...");
}

void loop() {
  int distance = ultrasound.Filter(); // Get filtered distance in mm
  Serial.println(distance);
  // Logic: If object is closer than 100mm, close the grip

  set_pose(0,180-(distance)/3,distance/3,30,30,90);
  // if (distance > 0 && distance < 100) {
  //   // Grasping Pose
  //   set_pose(0, 120, 0, 0, 0, 90);
  //   rgbs[0] = CRGB::Red;
  // } else {
  //   // Resting/Open Pose
  //   set_pose(90, 90, 0, 90, 90, 90);
  //   rgbs[0] = CRGB::Green;
  // }

  FastLED.show();
  servo_control(); // Smoothly move servos to the target pose
  delay(20); 
}

// Function to update the target pose
void set_pose(uint8_t s0, uint8_t s1, uint8_t s2, uint8_t s3, uint8_t s4, uint8_t s5) {
  target_angles[0] = s0;
  target_angles[1] = s1;
  target_angles[2] = s2;
  target_angles[3] = s3;
  target_angles[4] = s4;
  target_angles[5] = s5;
}

// Smooth servo interpolation logic from original source
void servo_control(void) {
  static uint32_t last_tick = 0;
  if (millis() - last_tick < 20) return;
  last_tick = millis();

  for (int i = 0; i < 6; ++i) {
    // Smooth movement logic (90% current + 10% target)
    if (servo_angles[i] > target_angles[i]) {
      servo_angles[i] = servo_angles[i] * 0.9 + target_angles[i] * 0.1;
    } else if (servo_angles[i] < target_angles[i]) {
      servo_angles[i] = servo_angles[i] * 0.9 + (target_angles[i] * 0.1 + 1);
    }

    // Constraint checking
    int16_t set_val = servo_angles[i];
    set_val = set_val < limt_angles[i][0] ? limt_angles[i][0] : set_val;
    set_val = set_val > limt_angles[i][1] ? limt_angles[i][1] : set_val;

    // Write to hardware
    servos[i].write(i == 0 || i == 5 ? 180 - set_val : set_val); 
  }
}