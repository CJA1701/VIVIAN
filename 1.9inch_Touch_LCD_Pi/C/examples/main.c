#include "test.h"
#include <math.h>
#include <stdlib.h>     //exit()
#include <stdio.h>

#include "DEV_Config.h"
#include "st7789.h"
#include "st7789.h"
#include "cst816d.h"


int main()
{
    touch_data_t touch_data;
    /* Module Init */
	if(DEV_ModuleInit() != 0){
        DEV_ModuleExit();
        exit(0);
    }
	
    st7789_init();
    cst816d_init();

    st7789_clear(0xF800);
    DEV_Delay_ms(1000);
    st7789_clear(0x07E0);
    DEV_Delay_ms(1000);
    st7789_clear(0x001F);
    DEV_Delay_ms(1000);
    st7789_clear(0xFFFF);
    DEV_Delay_ms(1000);

    while (1){
        if (get_touch_data(&touch_data)) {
            // printf("x: %d, y: %d \r\n", touch_data.coords[0].x, touch_data.coords[0].y);
            st7789_draw_rectangle(touch_data.coords[0].x, touch_data.coords[0].y, touch_data.coords[0].x + 4, touch_data.coords[0].y + 4, 0XF800);
        }
        DEV_Delay_ms(10);
    }
    
    return 0;
}
