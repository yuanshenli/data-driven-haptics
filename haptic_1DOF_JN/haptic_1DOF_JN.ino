/* PID setup */
#include <PID_v1_float_micros.h>
float Setpoint, Input, Output;
float Kp = 400, Ki = 5;
float Kd = 0.1 * Kp;
PID myPID(&Input, &Output, &Setpoint, Kp, Ki, Kd, DIRECT);
float lastSetpoint = Setpoint;

/* Gain scheduling setup */
float Kp_max = 400;
float Kp_min = 50;
float Kp_slope = -10;
float Kd_percent_Kp = 0.5;

/* Encoder setup */
#define SS_PIN 10
#include "SPI.h"
unsigned int clockSpeed = 5000000; //5 MHz
SPISettings AS5047_settings(clockSpeed, MSBFIRST, SPI_MODE1);
uint8_t byte1, byte2;
uint16_t rawAngle;
volatile long encoderValue = 0;
long lastencoderValue = 0;

/* filter setup */
const int filterWindowSize = 20;
long filterWindow[filterWindowSize];
int filterIndex = 0;
long rawAngleFiltered;

///* filter setup */
//const int outputFilterWindowSize = 1;
//long outputFilterWindow[filterWindowSize];
//int outputFilterIndex = 0;
//long outputFiltered;

/* Motor setup */
#define pwmPin0 2
#define pwmPin1 3
#define enablePin0 4
int pwmVal0 = 0;
int pwmVal1 = 0;
int pwmVal0_last = 0;
int pwmVal1_last = 0;
bool motorEnable = true;

/* Button setup */
#include <Bounce2.h>
#define buttonPin 23
Bounce debouncer = Bounce();

int rawForce = 0;

/* Serial setup */
String inMessage;
char t;

void setup() {
  Serial.begin(115200);

  /* PID setup */
  Input = 0;
  Setpoint = 150;
  myPID.SetMode(AUTOMATIC);
  int myRes = 12;
  myPID.SetOutputLimits(-pow(2, myRes), pow(2, myRes));
  myPID.SetSampleTime(100);   // Sample time 100 micros
  analogWriteResolution(myRes);
  /* Encoder setup */
  pinMode (SS_PIN, OUTPUT);
  digitalWrite(SS_PIN, HIGH);
  SPI.begin();
  /* Motor setup */
  pinMode(pwmPin0, OUTPUT);
  pinMode(pwmPin1, OUTPUT);
  analogWriteFrequency(pwmPin0, 14648.437);
  analogWriteFrequency(pwmPin1, 14648.437);
  pinMode(enablePin0, OUTPUT);
  digitalWrite(enablePin0, HIGH);
  /* Button setup */
  debouncer.attach(buttonPin, INPUT_PULLUP);
  debouncer.interval(25);
  for (int i = 0; i < filterWindowSize; i++) {
    updateEncoder();
    filterWindow[i] = rawAngle;
  }

  
}


void loop() {

  toggleMotorOnOff();
  readPot();
  readSerial();
  updateEncoder();
  filterEncoder();
  updateRawForce();
  
  if (myPID.Compute()) {
//    offsetOutput(1500.0, 4096.0);
//    calculateGain();
    myPID.SetTunings(Kp, Ki, Kd);
    // position control
//    pwmVal0 = (abs(Output) + Output) / 2;
//    pwmVal1 = (abs(Output) - Output) / 2;
    // force control
        pwmVal0 = (abs(Output) - Output) / 2;
        pwmVal1 = (abs(Output) + Output) / 2;

    analogWrite(pwmPin0, pwmVal0);
    analogWrite(pwmPin1, pwmVal1);
  }
  if (lastSetpoint != Setpoint) lastSetpoint = Setpoint;
  printVals();
}

void readSerial() {
  if (Serial.available()) {
    Kp_max = Serial.parseFloat();
    Kp_slope = Serial.parseFloat();
    Setpoint = Serial.parseFloat();
    Serial.print(Kp_max);
    Serial.print(", ");
    Serial.print(Kp_slope);
    Serial.print(", ");
    Serial.println(Setpoint);

  }
  while (Serial.available() > 0) t = Serial.read();
}

void readPot() {
  float potVal = analogRead(34);
//  Setpoint = potVal / 1024.0 * (280.0 - 70.0) + 70.0;
 Setpoint = potVal / 1024.0 * 50.0;
//  Serial.println(potVal);
}



void offsetOutput(float ofst, float outputMax) {
  Output = Output / outputMax * (outputMax - ofst) + Output / abs(Output) * ofst;
//  float turningPoint = 520;
//  if (abs(Output) < turningPoint) {
//    Output = (abs(Output) - ofst) / (turningPoint - ofst) * turningPoint * Output / abs(Output);
//  }
  
  
}

void calculateGain() {
  Kp = abs(Input - Setpoint) * Kp_slope + Kp_max;
  if (Kp < Kp_min) Kp = Kp_min;
  Kd = Kd_percent_Kp * Kp;
}


void toggleMotorOnOff() {
  debouncer.update();
  if ( debouncer.fell() ) {
    motorEnable = !motorEnable;
    digitalWrite(enablePin0, motorEnable);
  }
}

void updateRawForce() {
  int val1 = analogRead(21);
  int val2 = analogRead(22);
  rawForce = val2 - val1;
  Input = rawForce;

}
void updateEncoder() {
  SPI.beginTransaction(AS5047_settings);
  digitalWrite (SS_PIN, LOW);
  // reading only, so data sent does not matter
  byte1 = SPI.transfer(0);
  byte2 = SPI.transfer(0);
  digitalWrite (SS_PIN, HIGH);
  SPI.endTransaction();

  byte1 = byte1 & B00111111; // ignore the first two bits received
  rawAngle = byte1;
  rawAngle = rawAngle << 8;
  rawAngle |= byte2;


}

void filterEncoder() {
  filterWindow[filterIndex] = rawAngle;
  long sum = 0;
  for (int i = 0; i < filterWindowSize; i++) {
    sum += filterWindow[i];
  }
  rawAngleFiltered = (float)sum / (float)filterWindowSize;
  filterIndex++;
  filterIndex %= filterWindowSize;

  encoderValue = rawAngleFiltered;
  Input = (float)(encoderValue) / 4.0 * 360.0 / 4096.0;
}

long lastPrintTime = millis();
long currPrintTime; 
long printTimeInterval = 100;

void printVals() {
//  if (pwmVal0 != pwmVal0_last || pwmVal1 != pwmVal1_last) {
    currPrintTime = millis();
    if (currPrintTime - lastPrintTime > printTimeInterval) {
      Serial.print(Setpoint - Input);
      Serial.print(", ");
      Serial.print(Kp);
      Serial.print(", ");
      Serial.print(Output);
      Serial.print(", ");
      Serial.print(Input);
      Serial.println();
      lastPrintTime = currPrintTime;
    }
    
   
//    delay(100);

    pwmVal0_last = pwmVal0;
    pwmVal1_last = pwmVal1;
//  }
}
