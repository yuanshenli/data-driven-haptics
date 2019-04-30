
/* PID setup */
#include <PID_v1_float_micros.h>
float Setpoint, Input, Output;
float Kp = 15, Ki = 0.12;
float Kd = 0.012 * Kp;
PID myPID(&Input, &Output, &Setpoint, Kp, Ki, Kd, DIRECT);

/* Encoder setup */
#define SS_PIN 10
#include "SPI.h"
unsigned int clockSpeed = 5000000; //5 MHz
SPISettings AS5047_settings(clockSpeed, MSBFIRST, SPI_MODE1);
uint8_t byte1, byte2;
uint16_t rawAngle;

float Input_pos;

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

/* Playback setup */
#include "profile.h"

bool startSample = false;

void setup() {
  Serial.begin(115200);

  /* PID setup */
  Input = 0;
  Setpoint = 0;
  myPID.SetMode(AUTOMATIC);
  int myRes = 12;
  myPID.SetOutputLimits(500, pow(2, myRes));
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
  while (!startSample) {
    printVals();
    debouncer.update();
    if ( debouncer.fell() ) startSample = true;
  }
  updateEncoderAB();
  filterEncoderAB();
  int thisForce = updateRawForce();
  Input = thisForce;

  Setpoint = calculateSetpoint();
  
  if (myPID.Compute()) {
    offsetOutput(800.0, 4096.0);  
    pwmVal0 = (abs(Output) - Output) / 2;
    pwmVal1 = (abs(Output) + Output) / 2;
    analogWrite(pwmPin0, pwmVal0);
    analogWrite(pwmPin1, pwmVal1);
  }
  
  printVals();
}


float calculateSetpoint() {
  float interp_out = 0;
  int profileSize = sizeof(profilePos) / sizeof(profilePos[0]);
  if (Input_pos <= profilePos[0]) {
    interp_out = profileForce[0];
    return interp_out;
  } else if (Input_pos >= profilePos[profileSize]) {
    interp_out = profileForce[profileSize];
    return interp_out;
  } else {
    for (int i = 0; i < profileSize; i++) {
      if (Input_pos > profilePos[i]) continue;
      else {
        interp_out = profileForce[i-1] + (Input_pos - profilePos[i-1]) / (profilePos[i] - profilePos[i-1]) * (profileForce[i] - profileForce[i-1]);
        return interp_out;
      }
    }
  }
  return -1;
  
}


void offsetOutput(float ofst, float outputMax) {
  Output = Output / outputMax * (outputMax - ofst) + Output / abs(Output) * ofst;
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
//  Input_pos = ((float) encoderValue) * 360.0 / 4096.0;
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
  Input_pos = ((float) rawAngleFiltered) * 360.0 / 4096.0;
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
