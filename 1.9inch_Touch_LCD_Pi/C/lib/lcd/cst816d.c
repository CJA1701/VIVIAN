#include "cst816d.h"


static bool cst816d_read_flag = true;
static bool cst816d_int_enable = true;


static void cst816d_read(uint8_t reg_addr, uint8_t *data, uint32_t length) {
    I2C_Read_nByte(reg_addr, data, length);
}

bool get_touch_data(touch_data_t *touch_data) {
  uint8_t buff[12] = {0};
  uint8_t touch_num = 0;

  if (cst816d_int_enable) {
    if (false == cst816d_read_flag) {
      // Serial.println("cst816d_read_flag is false");
      return false;
    } else {
      cst816d_read_flag = false;
    }
  }

  cst816d_read(CST816D_TOUCH_NUM_REG, &touch_num, 1);
  touch_data->touch_num = touch_num;
  
  if (touch_num == 0)
    return false;


  cst816d_read(CST816D_TOUCH_XH_REG, &buff[0], 1);
  cst816d_read(CST816D_TOUCH_XL_REG, &buff[1], 1);
  cst816d_read(CST816D_TOUCH_YH_REG, &buff[2], 1);
  cst816d_read(CST816D_TOUCH_YL_REG, &buff[3], 1);



  for (uint16_t i = 0; i < touch_num; i++) {
    touch_data->coords[i].x = (((uint16_t)buff[(i * 6) + 0] & 0x0f) << 8) + buff[(i * 6) + 1];
    touch_data->coords[i].y = (((uint16_t)buff[(i * 6) + 2] & 0x0f) << 8) + buff[(i * 6) + 3];
  }

  return true;
  // cst816d_read(CST816D_TOUCH_XH_REG, buff, 1);
  // for (int i = 0; i < 6; i++)
  // {
  //   printf("%x ", buff[i]);
  // }
  // printf("\r\n");
  // for (uint16_t i = 0; i < touch_num; i++) {
  //   touch_data->coords[i].x = (((uint16_t)buff[(i * 6) + 0] & 0x0f) << 8) + buff[(i * 6) + 1];
  //   touch_data->coords[i].y = (((uint16_t)buff[(i * 6) + 2] & 0x0f) << 8) + buff[(i * 6) + 3];
  //   // printf("i: %d, x:%d, y:%d\r\n",, touch_data->coords[i].x, touch_data->coords[i].y);
  // }
  // return true;
}


// 中断服务函数
void cst816d_touch_int_cb(void) {
  cst816d_read_flag = true;
}


void cst816d_touch_rst(void)
{
    TP_RST_0;
    DEV_Delay_ms(100);
    TP_RST_1;
    DEV_Delay_ms(200);
}

void cst816d_init(void)
{
    uint8_t id = 0;
    cst816d_touch_rst();

    cst816d_read(CST816D_ID_REG, &id, 1);
    if (0 != id)
    {
        DEBUG("cst816d id:%x\r\n", id);
    }
    wiringPiISR(TP_INT,INT_EDGE_FALLING,&cst816d_touch_int_cb);
    cst816d_int_enable = true;
}

