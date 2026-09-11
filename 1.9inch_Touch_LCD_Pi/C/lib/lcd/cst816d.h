#ifndef __CST816D_H__
#define __CST816D_H__

#include "DEV_Config.h"
#include <stdint.h>

#include <stdlib.h>		//itoa()
#include <stdio.h>

#include "Debug.h"

#include "stdbool.h"


#define CST816D_ADDR 0x15
#define CST816D_ID_REG 0xA7

#define CST816D_TOUCH_NUM_REG 0X02
#define CST816D_TOUCH_XH_REG 0x03
#define CST816D_TOUCH_XL_REG 0x04
#define CST816D_TOUCH_YH_REG 0x05
#define CST816D_TOUCH_YL_REG 0x06


#define CST816D_TOUCH_MAX_NUM   1



typedef struct{
    uint16_t x;
    uint16_t y;
}touch_coords_t;

typedef struct{
  touch_coords_t coords[CST816D_TOUCH_MAX_NUM];
  uint16_t touch_num;
}touch_data_t;

void cst816d_init(void);

bool get_touch_data(touch_data_t *touch_data);


#endif  // __CST816D_H__
