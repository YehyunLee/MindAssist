/* 
 * 2024/02/21 hiwonder
 * 声波抓取例程(ultrasonic grasping program)
 * 超声波传感器检测到前方4cm左右的物体，会使用机械臂将物体夹取放置在左侧(If the ultrasonic sensor detects an object about 4cm ahead, the robotic gripper will grasp the object to place to the left side)
 */
#include <Wire.h>
#include <FastLED.h> //导入LED库(import LED library)
#include <Servo.h> //导入舵机库(import servo library)
#include "tone.h" //音调库(import tone library)
#include "Ultrasound.h" //导入超声波库(import ultrasonic library)
#include "mini_servo.h"

const static uint16_t DOC5[] = { TONE_C5 };
const static uint16_t DOC6[] = { TONE_C6 };

/* 引脚定义(define pins) */
const static uint8_t servoPins[5] = { 7, 6, 5, 4, 3};//舵机引脚定义(define servo pins)
const static uint8_t buzzerPin = 11;
const static uint8_t rgbPin = 13;

//RGB灯控制对象(RGB LED control object)
static CRGB rgbs[1];
//动作组控制对象(action group control object)
HW_ACTION_CTL action_ctl;
//舵机控制对象(servo control object)
Servo servos[5];
// 创建超声波对象(create ultrasonic object)
Ultrasound ul;

const uint8_t limt_angles[5][2] = {{0,82},{0,180},{0,180},{0,180},{0,180}}; /* 各个关节角度的限制(angle limit for each joint) */
static float servo_angles[5] = { 41 ,12 ,174 ,68 ,84 };  /* 舵机实际控制的角度数值(angle values for actual servo control) */

/* 蜂鸣器控制相关变量(variable related to buzzer control) */
static uint16_t tune_num = 0;
static uint32_t tune_beat = 10;
static uint16_t *tune;

static void servo_control(void);  //Servo control
void play_tune(uint16_t *p, uint32_t beat, uint16_t len); /* Buzzer control interface */
void tune_task(void); /* Buzzer control task */
static void ultrasound_task(void);  // Ultrasonic task

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  delay(100);
  Wire.setWireTimeout(25000, true); // 25ms I2C timeout, auto-reset bus on hang
  Serial.println("[DBG] serial ok");
  // 設置串行端口读取数据的超时时间(set the timeout for serial port reading data)
  Serial.setTimeout(500);

  // 绑定舵机IO口(bind servo IO pin)
  for (int i = 0; i < 5; ++i) {
    servos[i].attach(servoPins[i],500,2500);
  }
  Serial.println("[DBG] servos ok");

  //RGB灯初始化并控制(initialize and control RGB LED)
  FastLED.addLeds<WS2812, rgbPin, GRB>(rgbs, 1);
  rgbs[0] = CRGB(0, 0, 100);
  FastLED.show();
  Serial.println("[DBG] led ok");

  //Read deviation value
  action_ctl.read_offset();
  Serial.println("[DBG] offset ok");

  //Initialize the buzzer and it makes a sound
  pinMode(buzzerPin, OUTPUT);
  tone(buzzerPin, 1000);
  delay(100);
  noTone(buzzerPin);
  Serial.println("[DBG] buzzer ok");

  //ul.Color(0,0,255,0,0,255); // commented out - hangs if sensor I2C not responding

  // I2C scan - find what devices are on the bus
  Serial.println("[DBG] I2C scan:");
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("  Found device at 0x");
      Serial.println(addr, HEX);
    }
  }
  Serial.println("[DBG] scan done");

  delay(500);
  Serial.println("start");
}

void loop() {
  //超声波任务(ultrasonic task)
  ultrasound_task();
  // 蜂鸣器鸣响任务(buzzer sound task)
  tune_task();
  // 舵机控制(servo control)
  servo_control();
  // 动作组运动任务(action group running task)
  action_ctl.action_task();

}

// Ultrasonic task
void ultrasound_task(void)
{
  static uint32_t last_ul_tick = 0;
  static uint8_t step = 0;
  static uint8_t act_num = 0;
  static uint32_t delay_count = 0;

  // Interval 100ms
  if (millis() - last_ul_tick < 100) {
    return;
  }
  last_ul_tick = millis();

  // Obtain ultrasonic distance
  int dis = ul.Filter();
  Serial.print("d=");
  Serial.print(dis);
  Serial.print(" s=");
  Serial.println(step);

  switch(step)
  {
    case 0:
      if(dis > 35 && dis < 100) //If the block is detected
      {
        rgbs[0].r = 250;
        rgbs[0].g = 0;
        rgbs[0].b = 0;
        FastLED.show();
        play_tune(DOC6, 300u, 1u);
        act_num = 1;
        step++;
      }else //If the block is not detected
      {
        rgbs[0].r = 0;
        rgbs[0].g = 0;
        rgbs[0].b = 0;
        FastLED.show();
      }
      break;
    case 1: //Wait for 1s to put the block in the specified place.
      delay_count++;
      if(delay_count > 10)
      {
        delay_count = 0;
        // Action group running
        Serial.println("[ACT] starting action group");
        action_ctl.action_set(act_num);
        act_num = 0;
        step++;
      }
      break;
    case 2: //Wait for the action status to change to 0 
      if(action_ctl.action_state_get() == 0)
      {
        step = 0;
      }
      break;
    default:
      step = 0;
      break;
  }  

}

// 舵机控制任务(servo contorl task)
void servo_control(void) {
  static uint32_t last_tick = 0;
  if (millis() - last_tick < 20) {
    return;
  }
  last_tick = millis();

  for (int i = 0; i < 5; ++i) {
    if(servo_angles[i] > action_ctl.extended_func_angles[i])
    {
      servo_angles[i] = servo_angles[i] * 0.9 + action_ctl.extended_func_angles[i] * 0.1;
      if(servo_angles[i] < action_ctl.extended_func_angles[i])
        servo_angles[i] = action_ctl.extended_func_angles[i];
    }else if(servo_angles[i] < action_ctl.extended_func_angles[i])
    {
      servo_angles[i] = servo_angles[i] * 0.9 + (action_ctl.extended_func_angles[i] * 0.1 + 1);
      if(servo_angles[i] > action_ctl.extended_func_angles[i])
        servo_angles[i] = action_ctl.extended_func_angles[i];
    }
    servo_angles[i] = servo_angles[i] < limt_angles[i][0] ? limt_angles[i][0] : servo_angles[i];
    servo_angles[i] = servo_angles[i] > limt_angles[i][1] ? limt_angles[i][1] : servo_angles[i];
    
    servos[i].write(i == 0 || i == 5 ? 180 - servo_angles[i] : servo_angles[i]);
  }
}

// 蜂鸣器鸣响任务(buzzer sound task)
void tune_task(void) {
  static uint32_t l_tune_beat = 0;
  static uint32_t last_tick = 0;
  // 若未到定时时间 且 响的次数跟上一次的一样(if it is not yet the scheduled time and the ringing duration is the same as the previous time)
  if (millis() - last_tick < l_tune_beat && tune_beat == l_tune_beat) {
    return;
  }
  l_tune_beat = tune_beat;
  last_tick = millis();
  if (tune_num > 0) {
    tune_num -= 1;
    tone(buzzerPin, *tune++);
  } else {
    noTone(buzzerPin);
    tune_beat = 10;
    l_tune_beat = 10;
  }
}

// 蜂鸣器控制接口(buzzer contorl interface)
void play_tune(uint16_t *p, uint32_t beat, uint16_t len) {
  tune = p;
  tune_beat = beat;
  tune_num = len;
}
