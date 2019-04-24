#define SS_PIN 10
#include "SPI.h"
unsigned int clockSpeed = 5000000; //5 MHz
SPISettings AS5047_settings(clockSpeed, MSBFIRST, SPI_MODE1);
uint8_t byte1, byte2;
uint16_t rawAngle;

// AS5047 volatile register defines
#define NOP 0x0000 // No operation
#define ERRFL 0x0001 // Error register
#define PROG 0x0003 // Programming register
#define DIAAGC 0x3FFC // Diagnostic and AGC
#define MAG 0x3FFD // CORDIC magnitude
#define ANGLEUNC 0x3FFE // Measured angle without dynamic angle error compensation
#define ANGLECOM 0x3FFF // Measured angle with dynamic angle error compensation

// Masks for diagnostic and AGC (DIAAGC) register
#define MAGL 1<<11
#define MAGH 1<<10
#define COF 1<<9
#define LF 1<<8
#define AGC 0xFF


void setup() {
  Serial.begin(115200);
  pinMode (SS_PIN, OUTPUT);
  digitalWrite(SS_PIN, HIGH);
  SPI.begin();

}

void loop() {
  updateEncoder();
  Serial.print(rawAngle);
  Serial.println();
  delay(50);
}

void disableDAEC() {
  SPI.beginTransaction(AS5047_settings);
  
}


void updateEncoder() {
  SPI.beginTransaction(AS5047_settings);
  digitalWrite (SS_PIN, LOW);
  // reading only, so data sent does not matter
//  byte1 = SPI.transfer(0);
//  byte2 = SPI.transfer(0);
  uint16_t bytesToSend = 0x3FFF;
//  Serial.println(bytesToSend, BIN);
  bytesToSend = setParityBit(bytesToSend);
//  Serial.println(bytesToSend, BIN);
  rawAngle = SPI.transfer16(bytesToSend);
  digitalWrite (SS_PIN, HIGH);
  SPI.endTransaction();
  rawAngle = rawAngle & 0b0011111111111111;
//  rawAngle &= (1 << 14 - 1);
//  byte1 = byte1 & B00111111; // ignore the first two bits received
//  rawAngle = byte1;
//  rawAngle = rawAngle << 8;
//  rawAngle |= byte2;
  
//  Serial.print(rawAngle/16);
//  Serial.println();
}

uint16_t setParityBit(unsigned int val) {
  uint16_t original_val = val;
  uint16_t  count = 0;
  while (val != 0) {
    count += (val & 1);
    val >>= 1;
  }
  return count % 2 == 0 ? original_val : (original_val | 1 << 15);
}
