/*
 * 2024/02/21 hiwonder CuZn
 * Arduino与ESP32Cam的IIC通讯类(the I2C communication class between Arduino and ESP32Cam)
 * 注意：每个不同的功能，ESP32Cam也要下载对应功能的ESP32Cam程序(Note: For each different function, ESP32Cam needs to be downloaded with corresponding ESP32Cam function)
 */

#ifndef __HW_ESP32CAM_CTL_H_
#define __HW_ESP32CAM_CTL_H_

#include <Arduino.h>
#include <Wire.h>

#define ESP32CAM_ADDR 0x52

class HW_ESP32Cam{
  public:
    //初始化IIC(initialize I2C)
    void begin(void);
    //人脸识别获取函数(face recognition obtaining function)
    bool faceDetect(void);
    //颜色识别获取函数(color recognition obtaining function)
    int colorDetect(void);
    // 颜色位置获取函数(color position obtaining function)
    bool color_position(uint16_t *color_info);

    void set_led(uint8_t lightness);

};

#endif //__ESP32CAM_CTL_H_

