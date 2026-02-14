#ifndef _HW_ACTION_CTL_
#define _HW_ACTION_CTL_
#include "actions.h"
#include <EEPROM.h>

#define EEPROM_START_FLAG "HIWONDER"
#define EEPROM_SERVO_OFFSET_START_ADDR 1024
#define EEPROM_SERVO_OFFSET_DATA_ADDR 1024+16
#define EEPROM_SERVO_OFFSET_LEN 6u

class HW_ACTION_CTL{
  public:
  /* the using angles of secondary development example */
    uint8_t extended_func_angles[5] = { 41 ,12 ,174 ,68 ,84 }; 
    //Control execution action group
    void action_set(int num);
    int action_state_get(void);
    void action_task(void);
    void read_offset();
    int8_t* get_offset(void);
    
  private:
    //Action group control variables
    int action_num = 0;
    int8_t servo_offset[5];
    uint8_t eeprom_read_buf[16];
};

#endif //_HW_ACTION_CTL_
