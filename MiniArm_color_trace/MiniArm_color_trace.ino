/* 
 * 2024/02/21 hiwonder CuZn
 * 颜色夹取例程(color sorting program)
 * 用IIC连接ESP32Cam，检测蓝色、绿色，分别夹取到两边(Connect ESP32S3-Cam with I2C to detect blue and red objects, grasping them to their respective sides)
 * 注意：每个不同的功能，ESP32Cam也要下载对应功能的ESP32Cam程序(Note: For each different function, the ESP32S3-Cam also needs to be downloaded with corresponding ESP32S3-Cam program)
 */
#include <FastLED.h> //导入LED库(import LED library)
#include <Servo.h> //导入舵机库(import servo library)
#include "hw_esp32cam_ctl.h" //导入ESP32Cam通讯库(import ESP32S3-Cam communication library)
#include <EEPROM.h>

#define EEPROM_START_FLAG "HIWONDER"
#define EEPROM_SERVO_OFFSET_START_ADDR 1024
#define EEPROM_SERVO_OFFSET_DATA_ADDR 1024+16
#define EEPROM_SERVO_OFFSET_LEN 6u

/* 引脚定义(define pins) */
const static uint8_t servoPins[5] = { 7, 6, 5, 4, 3 };
const static uint8_t buzzerPin = 11;
const static uint8_t rgbPin = 13;

//RGB灯控制对象(RGB light control object)
static CRGB rgbs[1];
//ESP32Cam通讯对象(ESP32Cam communication object)
HW_ESP32Cam hw_cam;
//舵机控制对象(servo control object)
Servo servos[5];

/* The used angle values of secondary development example */
static uint8_t extended_func_angles[5] = { 90,115,180,120,90 }; 
/* Limit of each joint’s angle */
const uint8_t limt_angles[5][2] = {{0,90},{0,180},{0,180},{25,180},{0,180} }; 
/* Actual angle values of servo control */
static uint8_t servo_angles[5] = { 90,115,180,120,90 };  
static int8_t servo_offset[5];
static uint8_t servo_expect[5];
static uint8_t eeprom_read_buf[16];

static void servo_control(void); /* servo control */
void espcam_task(void); /* esp32cam communication task */
static void read_offset(void);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  // set the timeout for reading data from the serial port
  Serial.setTimeout(500);

  // 绑定舵机IO口(bind servo IO port)
  for (int i = 0; i < 5; ++i) {
    servos[i].attach(servoPins[i]);
  }

  hw_cam.begin(); //initialize the ESP32Cam communication connector
  hw_cam.set_led(10); // control ESP32Cam fill light brightness [brightness value (0~255)]

  //RGB灯初始化并控制(initialize and control the RGB light)
  FastLED.addLeds<WS2812, rgbPin, GRB>(rgbs, 1);
  rgbs[0] = CRGB(0, 0, 100);
  FastLED.show();

  read_offset();

  //蜂鸣器初始化并鸣响一声(initialize and control the buzzer to make a sound)
  pinMode(buzzerPin, OUTPUT);
  tone(buzzerPin, 1000);
  delay(100);
  noTone(buzzerPin);

  delay(2000);
  Serial.println("start");
}

void loop() {
  // esp32cam communication task
  espcam_task();
  // servo control
  servo_control();
}

// esp32cam通讯任务(ESP32S3-Cam communication task)
void espcam_task(void)
{
  static uint32_t last_tick = 0;
  uint16_t color_info[4];

  // time intervals
  if (millis() - last_tick < 75) {
    return;
  }
  last_tick = millis();
  
  if(hw_cam.color_position(color_info)) //if the color is recognized
  {
    uint16_t num = color_info[0] + color_info[2]/2; //calculate the center point of the color block
    uint16_t angle = map(num , 0 , 320 , 60 , 120); //Map to the corresponding servo angle
    extended_func_angles[4] = angle; ///control servo’s movement tracking
  }
}


// 舵机控制任务（不需修改）(servo contorl task (no need to modify))
void servo_control(void) {
  static uint32_t last_tick = 0;
  if (millis() - last_tick < 20) {
    return;
  }
  last_tick = millis();
  for (int i = 0; i < 5; ++i) {
    servo_expect[i] = extended_func_angles[i] + servo_offset[i];
    if(servo_angles[i] > servo_expect[i])
    {
      servo_angles[i] = servo_angles[i] * 0.9 + servo_expect[i] * 0.1;
      if(servo_angles[i] < servo_expect[i])
        servo_angles[i] = servo_expect[i];
    }else if(servo_angles[i] < servo_expect[i])
    {
      servo_angles[i] = servo_angles[i] * 0.9 + (servo_expect[i] * 0.1 + 1);
      if(servo_angles[i] > servo_expect[i])
        servo_angles[i] = servo_expect[i];
    }

    servo_angles[i] = servo_angles[i] < limt_angles[i][0] ? limt_angles[i][0] : servo_angles[i];
    servo_angles[i] = servo_angles[i] > limt_angles[i][1] ? limt_angles[i][1] : servo_angles[i];
    servos[i].write(i == 0 || i == 5 ? 180 - servo_angles[i] : servo_angles[i]);
  }
}

void read_offset(void){
  // 读取偏差角度(read deviation angle) 
  for (int i = 0; i < 16; ++i) {
    eeprom_read_buf[i] = EEPROM.read(EEPROM_SERVO_OFFSET_START_ADDR + i);
  }
  if (strcmp(eeprom_read_buf, EEPROM_START_FLAG) == 0) {
    memset(eeprom_read_buf,  0 , sizeof(eeprom_read_buf));
    Serial.println("read offset");
    for (int i = 0; i < 5; ++i) {
      eeprom_read_buf[i] = EEPROM.read(EEPROM_SERVO_OFFSET_DATA_ADDR + i);
    }
    memcpy(servo_offset , eeprom_read_buf , 6);
    /*for (int i = 0; i < 5; ++i) {
      Serial.print(servo_offset[i]);
      Serial.print(" ");
    }
    Serial.println("");*/
  }
}


