
/* PID setup */
#include <PID_v1_float_micros.h>
float Setpoint, Input, Output;
float Kp = 75;
float Ki = 0.75;
float Kd = 250; //  * Kp;
PID myPID(&Input, &Output, &Setpoint, Kp, Ki, Kd, DIRECT);

/* Gain scheduling setup */
float Kp_max = 150;
float Kp_min = 50;
float Kp_slope = -5;
float Kd_percent_Kp = 0.015;

/* Encoder setup */
#define SS_PIN 10
#include "SPI.h"
unsigned int clockSpeed = 5000000; //5 MHz
SPISettings AS5047_settings(clockSpeed, MSBFIRST, SPI_MODE1);
uint8_t byte1, byte2;
uint16_t rawAngle;

/* filter setup */
const int filterWindowSize = 20;
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
bool motorEnable = true;

/* Button setup */
#include <Bounce2.h>
#define buttonPin 23
Bounce debouncer = Bounce();

/*Print setup*/
long lastPrintTime = millis();
long currPrintTime; 
long printTimeInterval = 10;

/* Serial setup */
String inMessage;
char t;

/* Sample setup */
const int bufSize = 2000;
float posBuffer[bufSize];
float forceBuffer[bufSize];
int bufCount = 0;

const int dataSize = 200;
const float posRes = 1.0;
const float posMax = 200.0;
const float posMin = 0.0;
float posData[dataSize];
float forceData[dataSize];
int dataCount = 0;

bool startSample = false;

void setup() {
  Serial.begin(115200);

  /* PID setup */
  Input = 0;
  Setpoint = posMin;
  myPID.SetMode(AUTOMATIC);
  int myRes = 12;
  myPID.SetOutputLimits(-pow(2, myRes), pow(2, myRes));
  myPID.SetSampleTime(100);   // Sample time 100 micros
  analogWriteResolution(myRes);
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
  analogWriteFrequency(pwmPin0, 18000);
  analogWriteFrequency(pwmPin1, 18000);
  pinMode(enablePin0, OUTPUT);
  digitalWrite(enablePin0, HIGH);
  /* Button setup */
  debouncer.attach(buttonPin, INPUT_PULLUP);
  debouncer.interval(25);
  for (int i = 0; i < filterWindowSize; i++) {
    filterWindow[i] = 0;
  }
}


void loop() {
  toggleState();
  updateEncoderAB();
  filterEncoderAB();
  int thisForce = updateRawForce();
  if (myPID.Compute()) {
    offsetOutput(800.0, 4096.0);  
    myPID.SetTunings(Kp, Ki, Kd);
    pwmVal0 = (abs(Output) - Output) / 2;
    pwmVal1 = (abs(Output) + Output) / 2;
    analogWrite(pwmPin0, pwmVal0);
    analogWrite(pwmPin1, pwmVal1);
  }
  
  
//  if (bufCount < bufSize) {
    posBuffer[bufCount] = Input;
    forceBuffer[bufCount] = thisForce;
    bufCount++;
//  }
  if (bufCount == bufSize) {
//    if (dataCount < dataSize) {
      posData[dataCount] = averageBuf(posBuffer, bufSize);
      forceData[dataCount] = averageBuf(forceBuffer, bufSize);
//      Serial.print(posData[dataCount]);
//      Serial.print(", ");
//      Serial.println(forceData[dataCount]);
      dataCount++;
      Setpoint += posRes;
//    }
    bufCount %= bufSize;
  }

  if (dataCount == dataSize) {
    Serial.print("pos = ");
    printBuf(posData, dataSize);
    Serial.print("force = ");
    printBuf(forceData, dataSize);
    analogWrite(pwmPin0, 0);
    analogWrite(pwmPin1, 0);
    while(true);
  }
  
//  printVals();
}

float averageBuf(float arr[], int  arrSize) {
  double sum = 0;
  for (int i = 0; i < arrSize; i++) {
    sum += arr[i];
  }
  return sum/(float)arrSize;
}

void printBuf(float arr[], int arrSize) {
  Serial.print("[");
  for (int i = 0; i < arrSize; i++) {
    Serial.print(arr[i]);
    if (i < arrSize - 1) {
      Serial.print(", ");
    } else {
      Serial.println("] ");
    }
  }
}

void toggleState() {
  debouncer.update();
  if ( debouncer.fell() ) startSample = !startSample;
  while (!startSample) {
    debouncer.update();
    if ( debouncer.fell() ) startSample = !startSample;
    currPrintTime = millis();
    if (currPrintTime - lastPrintTime > printTimeInterval) {
      Serial.println("-1");
      lastPrintTime = currPrintTime;
    }
  }
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


void offsetOutput(float ofst, float outputMax) {
  Output = Output / outputMax * (outputMax - ofst) + Output / abs(Output) * ofst;
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

int updateRawForce() {
  int val2 = analogRead(22);
  return val2;
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
//  Input = ((float) encoderValue) * 360.0 / 4096.0;
}

void updateEncoderI() {
  encoderValue = 0;
}

void filterEncoderAB() {
  filterWindow[filterIndex] = encoderValue;
  long sum = 0;
  for (int i = 0; i < filterWindowSize; i++) {
    sum += filterWindow[i];
  }
  rawAngleFiltered = (float)sum / (float)filterWindowSize;
  filterIndex++;
  filterIndex %= filterWindowSize;
  Input = ((float) rawAngleFiltered) * 360.0 / 4096.0;
}

void printVals() {
    currPrintTime = millis();
    if (currPrintTime - lastPrintTime > printTimeInterval) {
      Serial.print(Setpoint);
      Serial.print(", ");
//      Serial.print(bufCount);
//      Serial.print(", ");
//      Serial.print(dataCount);
//      Serial.print(", ");
      Serial.print(Input);
      Serial.println();
      lastPrintTime = currPrintTime;
    }
}
