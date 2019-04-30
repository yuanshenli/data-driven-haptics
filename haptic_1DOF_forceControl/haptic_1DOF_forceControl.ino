#define FORCE_CONTROL

/* PID setup */
#include <PID_v1_float_micros.h>
float Setpoint, Input, Output;

#ifdef FORCE_CONTROL
float Kp = 15, Ki = 0.12;
float Kd = 0.012 * Kp;
bool forceControl = true;
#else
//float Kp = 300;
//float Ki = 1.5;
//float Kd = 0.3 * Kp;
float Kp = 0, Ki = 0;
float Kd = 2500.0;
bool forceControl = false;
#endif

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

/* filter setup */
const int filterWindowSize = 150;
long filterWindow[filterWindowSize];
int filterIndex = 0;
float rawAngleFiltered;

/* Encoder ABI setup */
#define ENC_A 7 // pin for ENC_A
#define ENC_B 6 // pin for ENC_B
#define ENC_I 5
volatile int lastEncoded = 0;
volatile long encoderValue = 0;

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
  Setpoint = 500;
  myPID.SetMode(AUTOMATIC);
  int myRes = 12;
  myPID.SetOutputLimits(500, pow(2, myRes));
  myPID.SetSampleTime(100);   // Sample time 100 micros
  analogWriteResolution(myRes);
  /* Encoder setup */
  pinMode (SS_PIN, OUTPUT);
  digitalWrite(SS_PIN, HIGH);
  SPI.begin();
  /* Encoder ABI setup */
  pinMode(ENC_A, INPUT);
  pinMode(ENC_B, INPUT);
  pinMode(ENC_I, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_A), updateEncoderAB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), updateEncoderAB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_I), updateEncoderI, RISING);
  /* Motor setup */
  pinMode(pwmPin0, OUTPUT);
  pinMode(pwmPin1, OUTPUT);
//  analogWriteFrequency(pwmPin0, 14648.437);
//  analogWriteFrequency(pwmPin1, 14648.437);
  analogWriteFrequency(pwmPin0, 18000);
  analogWriteFrequency(pwmPin1, 18000);
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
  if (forceControl) {
    updateRawForce();
  } else {
//    updateEncoder();
  }
//  filterEncoderAB();
  filterEncoder();
  
  
  if (myPID.Compute()) {
//    offsetOutput(1500.0, 4096.0);
//  if (!forceControl) {
//    calculateGain();
//  }
//   
    myPID.SetTunings(Kp, Ki, Kd);
    pwmVal0 = 0;
    pwmVal1 = Output;
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
  if (forceControl) {
    Setpoint = potVal / 1024.0 * 300.0;
  } else {
//    Setpoint = potVal / 1024.0 * (280.0 - 70.0) + 70.0;
    Setpoint = potVal / 1024.0 * 200.0;
  }
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
  int val2 = analogRead(22);
  rawForce = val2;
//  Input = rawForce;
}

void updateEncoderAB() {
  int MSB = digitalRead(ENC_A); //MSB = most significant bit
  int LSB = digitalRead(ENC_B); //LSB = least significant bit
  int encoded = (MSB << 1) | LSB; //converting the 2 pin value to single number
  int sum  = (lastEncoded << 2) | encoded; //adding it to the previous encoded value
  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValue ++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValue --;
  lastEncoded = encoded; //store this value for next time
  encoderValue %= 4096;
  rawAngle = encoderValue;
  Input = ((float) encoderValue) * 360.0 / 4096.0;
  
}

void updateEncoderI() {
  encoderValue = 0;
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
  if (forceControl) filterWindow[filterIndex] = rawForce;
  else {
    filterWindow[filterIndex] = rawAngle;
  }
  
  long sum = 0;
  for (int i = 0; i < filterWindowSize; i++) {
    sum += filterWindow[i];
  }
  rawAngleFiltered = (float)sum / (float)filterWindowSize;
  filterIndex++;
  filterIndex %= filterWindowSize;

  if (forceControl) {
    Input = rawAngleFiltered;
  }
  else {
    Input = (float)(rawAngleFiltered) / 4.0 * 360.0 / 4096.0;
  }
}

void filterEncoderAB() {
  if (forceControl) filterWindow[filterIndex] = rawForce;
  else {
    filterWindow[filterIndex] = encoderValue;
  }
  
  long sum = 0;
  for (int i = 0; i < filterWindowSize; i++) {
    sum += filterWindow[i];
  }
  rawAngleFiltered = (float)sum / (float)filterWindowSize;
  filterIndex++;
  filterIndex %= filterWindowSize;

  if (forceControl) {
    Input = rawAngleFiltered;
  }
  else {
//      encoderValue = rawAngleFiltered;
//    Input = (float)(rawAngleFiltered) / 4.0 * 360.0 / 4096.0;
     Input = ((float) rawAngleFiltered) * 360.0 / 4096.0;
  }
}

long lastPrintTime = millis();
long currPrintTime; 
long printTimeInterval = 10;

void printVals() {
    currPrintTime = millis();
    if (currPrintTime - lastPrintTime > printTimeInterval) {
      Serial.print(Setpoint);
      Serial.print(", ");
//      Serial.print(encoderValue);
//      Serial.print(", ");
//      Serial.print(Output);
//      Serial.print(", ");
      Serial.print(Input);
      Serial.println();
      lastPrintTime = currPrintTime;
    }
    pwmVal0_last = pwmVal0;
    pwmVal1_last = pwmVal1;
}
