#include "st7789.h"

static void st7789_write_bytes(uint8_t *data, uint32_t lenght)
{
    LCD_DC_1;
    DEV_SPI_Write_nByte(data, lenght);
}

static void st7789_write_byte(uint8_t data)
{
    LCD_DC_1;
    //LCD_2IN8_CS_0;
    DEV_SPI_WriteByte(data);
    //LCD_2IN8_CS_1;
}

static void st7789_write_cmd(uint8_t cmd)
{
    LCD_DC_0;
    //LCD_2IN8_CS_0;
    DEV_SPI_WriteByte(cmd);
    //LCD_2IN8_CS_1;
}


static void st7789_rst(void)
{
    LCD_RST_1;
    DEV_Delay_ms(100);
    LCD_RST_0;
    DEV_Delay_ms(100);
    LCD_RST_1;
    DEV_Delay_ms(100);
}



void st7789_init(void)
{
    st7789_rst();

    st7789_write_cmd(0x11);     

    DEV_Delay_ms(120);                //DEV_Delay_ms 120ms

    st7789_write_cmd(0x36);     // Memory Data Access Control MY,MX~~
    st7789_write_byte(0x48);   

    st7789_write_cmd(0x3A);     
    st7789_write_byte(0x05);   //st7789_write_byte(0x66);

    st7789_write_cmd(0xF0);     // Command Set Control
    st7789_write_byte(0xC3);   

    st7789_write_cmd(0xF0);     
    st7789_write_byte(0x96);   

    st7789_write_cmd(0xB4);     
    st7789_write_byte(0x01);   

    st7789_write_cmd(0xB7);     
    st7789_write_byte(0xC6);   

    st7789_write_cmd(0xC0);     
    st7789_write_byte(0x80);   
    st7789_write_byte(0x45);   

    st7789_write_cmd(0xC1);     
    st7789_write_byte(0x13);   //18  //00

    st7789_write_cmd(0xC2);     
    st7789_write_byte(0xA7);   

    st7789_write_cmd(0xC5);     
    st7789_write_byte(0x0A);   


    st7789_write_cmd(0xF0);     
    st7789_write_byte(0x3C);   

    st7789_write_cmd(0xF0);     
    st7789_write_byte(0x69);   

    DEV_Delay_ms(120);                

    st7789_write_cmd(0x21);     

    st7789_write_cmd(0x29); 

}


void st7789_set_windows(uint16_t x_start, uint16_t y_start, uint16_t x_end, uint16_t y_end)
{
    uint8_t buffer[4];
    x_start += 35;
    x_end += 35;


    buffer[0] = x_start >> 8;
    buffer[1] = (uint8_t)(x_start & 0x00ff);
    buffer[2] = x_end >> 8;
    buffer[3] = (uint8_t)(x_end & 0x00ff);
    st7789_write_cmd(0x2a);
    st7789_write_bytes(buffer, 4);

    buffer[0] = y_start >> 8;
    buffer[1] = (uint8_t)(y_start & 0x00ff);
    buffer[2] = y_end >> 8;
    buffer[3] = (uint8_t)(y_end & 0x00ff);
    st7789_write_cmd(0x2b);
    st7789_write_bytes(buffer, 4);
    st7789_write_cmd(0x2c);
}


void st7789_draw_rectangle(uint16_t x_start, uint16_t y_start, uint16_t x_end, uint16_t y_end, uint16_t color)
{
    uint8_t color_h = color >> 8;
    uint8_t color_l = (uint8_t)(color && 0x00ff);
    uint16_t i = 0;
    uint16_t color_lenght = (1 + x_end - x_start) * (1 + y_end - y_start);
    st7789_set_windows(x_start, y_start, x_end, y_end);
    for (i = 0; i < color_lenght; i++){
        st7789_write_byte(color_h);
        st7789_write_byte(color_l);
    }
}

void st7789_flush(uint16_t x_start, uint16_t y_start, uint16_t x_end, uint16_t y_end, uint16_t *color)
{

}

uint16_t swap_uint16(uint16_t val) {
    return (val << 8) | (val >> 8);
}

void st7789_clear(uint16_t color)
{
    
    uint16_t buf[ST7789_WIDTH];
    uint16_t i;
    color = swap_uint16(color);
    for (i = 0; i < ST7789_WIDTH; i++)
        buf[i] = color;

    st7789_set_windows(0, 0, ST7789_WIDTH - 1, ST7789_HEIGHT - 1);

    for (i = 0; i < ST7789_HEIGHT; i++)
        st7789_write_bytes((uint8_t *)buf ,ST7789_WIDTH * 2);
}
