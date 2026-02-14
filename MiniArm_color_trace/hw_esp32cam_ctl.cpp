/* 
 * 2024/02/21 hiwonder CuZn
 * Arduino与ESP32Cam的IIC通讯类(the I2C communication class between Arduino and ESP32Cam)
 */
#include "hw_esp32cam_ctl.h"

void HW_ESP32Cam::begin(void)
{
  Wire.begin();
}

//写多个字节(write multiple bytes)
static bool wireWriteDataArray(uint8_t addr, uint8_t reg,uint8_t *val,unsigned int len)
{
    unsigned int i;

    Wire.beginTransmission(addr);
    Wire.write(reg);
    for(i = 0; i < len; i++) 
    {
        Wire.write(val[i]);
    }
    if( Wire.endTransmission() != 0 ) 
    {
        return false;
    }
    return true;
}

static bool WireWriteByte(uint8_t val)
{
    Wire.beginTransmission(ESP32CAM_ADDR);
    Wire.write(val);
    if( Wire.endTransmission() != 0 ) {
        return false;
    }
    return true;
}

static int WireReadDataArray(uint8_t reg, uint8_t *val, unsigned int len)
{
    unsigned char i = 0;
    
    /* Indicate which register we want to read from */
    if (!WireWriteByte(reg)) {
        return -1;
    }
    
    /* Read block data */
    Wire.requestFrom(ESP32CAM_ADDR, len);
    while (Wire.available()) {
        if (i >= len) {
            return -1;
        }
        val[i] = Wire.read();
        i++;
    }   
    return i;
}

//读取ESP32Cam检测人脸(read ESP32Cam detected face)
bool HW_ESP32Cam::faceDetect(void)
{
  uint8_t face_info[4];
  Serial.print("face ");
  int num = WireReadDataArray(0x01,face_info,4);
  if((num == 4) && (face_info[2] > 0)) //接收识别到的人脸的x,y,w,h值(receive x, y, w, and h values of the detected face)
  {
      Serial.println(" 1");
      return true;
  }
  Serial.println(" 0");
  return false;
}

//读取ESP32Cam识别颜色，返回颜色代号(read ESP32Cam recognized color and return color number)
int HW_ESP32Cam::colorDetect(void)
{
  Serial.print("colorDetect");
  uint8_t color_info[2][4];
  int num = WireReadDataArray(0x00,color_info[0],4);
  if((num == 4) && (color_info[0][2] > 0)) //接收识别到的颜色的x,y,w,h值(receive x, y, w, and h values of the recognized color)
  {
      return 2;
  }
  num = WireReadDataArray(0x01,color_info[1],4);
  if(num == 4)
  {
    if(color_info[1][2] > 0) //若w值大于0，则识别到颜色1(If the w value is greater than 0, the color 1 is recognized)
    {
      return 1;
    }
  }
  return 0;
}

//读取ESP32Cam识别颜色位置，读取成功返回true和位置数据(read the recognied color position of ESP32Cam, and return true and position data if successfully read)
bool HW_ESP32Cam::color_position(uint16_t *color_info)
{
  Serial.print("color_position");
  int num = WireReadDataArray(0x01,(uint8_t*)color_info,8);
  if((num == 8) && (color_info[2] > 0)) //接收识别到的颜色的x,y,w,h值(receive x, y, w, and h values of the recognized color)
  {
    return true;
  }
  return false;
}

void HW_ESP32Cam::set_led(uint8_t lightness)
{
  lightness = lightness < 0 ? 0 : lightness;
  lightness = lightness > 255 ? 255 : lightness;
  wireWriteDataArray(ESP32CAM_ADDR , 0x11 , &lightness , 1);
}
