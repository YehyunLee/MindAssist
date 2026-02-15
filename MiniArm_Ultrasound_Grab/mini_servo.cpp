#include "mini_servo.h"

void HW_ACTION_CTL::action_set(int num){
  action_num = num;
}

int HW_ACTION_CTL::action_state_get(void){
  return action_num;
}

void HW_ACTION_CTL::read_offset(void){
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

int8_t* HW_ACTION_CTL::get_offset(void){
  return servo_offset;
}

void HW_ACTION_CTL::action_task(void){
  static uint32_t last_tick = 0;
  static uint8_t step = 0;
  static uint8_t num = 0 , delay_count = 0;
  if(action_num != 0 && action_num <= action_count)
  {
    // 间隔时间(time interval)
    if (millis() - last_tick < 100) {
      return;
    }
    last_tick = millis();
    
    switch(step)
    {
      case 0: //运行动作(run action)
        if(action[action_num-1][num][0] != 0)
        {
          extended_func_angles[0] = action[action_num-1][num][1] + servo_offset[0];
          extended_func_angles[1] = action[action_num-1][num][2] + servo_offset[1];
          extended_func_angles[2] = action[action_num-1][num][3] + servo_offset[2];
          extended_func_angles[3] = action[action_num-1][num][4] + servo_offset[3];
          extended_func_angles[4] = action[action_num-1][num][5] + servo_offset[4];

          step = 1;

        }else{ //若运行完毕(if the running is completed)
          num = 0;
          // 清空动作组变量(clear action group variable)
          action_num = 0;
          
        }
        break;
      case 1: //等待动作运行(wait for the action to be executed)
        delay_count++;
        if(delay_count > 5)
        {
          num++;
          delay_count = 0;
          step = 0;
        }
        break;
      default:
        step = 0;
        break;
    }
  }
}