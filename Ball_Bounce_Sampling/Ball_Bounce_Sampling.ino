
#include <DDHapticHelper.h>
float myKp = 75;
float myKi = 0.75;
float myKd = 250; //  * Kp;

bool isSaturated = false;
float lastPos = 0;
float lastForce = 0;
float maxForce = 600;
float forceIncFill = 20;
float posIncFill = posRes;

#define buttonPin2 37
Bounce debouncer2 = Bounce();

#define accPinZ A19
#define accPinY A20
#define accPinX A21
int accZ = 0;
int accY = 0;
int accX = 0;

int thisForce = 0;

typedef enum {
  WAIT,
  SAMPLE,
  PRINT
} States;

typedef enum {
  SAMPLE_BACK,
  SAMPLE_FORWARD
} Sample_States;

States currState = WAIT;
Sample_States currSampleState = SAMPLE_BACK;

/* sampling profile */
float A_samp = 220.0;
int fs = 1000;        // [Hz]
long t_samp = 0;
unsigned int t_ms_curr;
unsigned int t_ms_last = 0;
int t_res = 1;            // [ms]

float f0 = 5.5;           // [Hz]
float f1 = 7.0;          // [Hz]
float sweepTime = 10.0;   // [s]

unsigned int T_samp_update_curr = 0;
unsigned int T_samp_update_last = 0;
//unsigned int T_samp_update_interval = ;


void setup() {
  Serial.begin(115200);

  /* PID setup */
  Input = 0;
  Setpoint = posMin;
  myPID.SetMode(AUTOMATIC);
  int myRes = 12;
  myPID.SetOutputLimits(-pow(2, myRes), pow(2, myRes));
  myPID.SetSampleTime(100);   // Sample time 100 micros
  myPID.SetTunings(myKp, myKi, myKd);
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
  debouncer2.attach(buttonPin2, INPUT_PULLUP);
  debouncer2.interval(25);
  for (int i = 0; i < filterWindowSize; i++) {
    filterWindow[i] = 0;
  }
}

int ct = 0;
void loop() {
  ct++;
  printVals();
  updateAcc();
  thisForce = updateRawForce();
  updateEncoderAB();
  filterEncoderAB();
  Input = Input_pos;

  switch (currState) {
    case WAIT:
      motorOff();
      debouncer.update();
      if (debouncer.fell()) {
        currState = SAMPLE;
        Serial.println("switch to SAMPLE_BACK");
      }
      break;
    case SAMPLE:
      //      manualSample();
      autoSample();

      break;
    case PRINT:
      motorOff();
      currState = WAIT;
      Serial.println("switch to WAIT");
      break;
  }


}

void autoSample() {
  debouncer.update();
  if (debouncer.fell()) {
    currState = PRINT;
    Serial.println("switch to PRINT");
  }
  if (myPID.Compute()) {
    offsetOutput(800.0, 4096.0);
    //      myPID.SetTunings(Kp, Ki, Kd);
    pwmVal0 = (abs(Output) - Output) / 2;
    pwmVal1 = (abs(Output) + Output) / 2;
    analogWrite(pwmPin0, pwmVal0);
    analogWrite(pwmPin1, pwmVal1);
  }
  //  T_samp = analogRead(34);
  t_ms_curr = millis();
  if (t_ms_curr - t_ms_last > 1000.0 / (float)fs) {
    t_ms_last = t_ms_curr;
    Setpoint = bidirChirp(A_samp, A_samp, f0, f1, sweepTime, t_samp, fs);
    t_samp++;
  }
}

/*************************************
   function: bidirChipr
   Inputs:
        float A_max: largest magnitude
        float A_min: smallest magnitude
        float f0: smallest freq
        float f1: largest freq
        float sweepTime: time from f0 to f1, and vice versa [s]
        int tn: time stamps for each sample
        int fs: sampling freq  [fq]
*/

float bidirChirp(float A_max, float A_min, float f0, float f1, float sweepTime, int tn, int fs) {
  
  int n_sample_per_sweep = (int)sweepTime * fs;
  int tn_mod = tn % (2 * n_sample_per_sweep);
  float slope = (f1 - f0) / (float) n_sample_per_sweep;

  if (tn_mod > n_sample_per_sweep) tn_mod = 2 * n_sample_per_sweep - tn_mod;
  float f_n = f0 + slope * (float) tn_mod;

  float y = A_max * sin(2.0 * M_PI * f_n * (float) tn_mod / (float) fs ) - A_max;
  return y;
}



void manualSample() {
  debouncer.update();
  if (debouncer.fell()) {
    currState = PRINT;
    Serial.println("switch to PRINT");
  }
  if (myPID.Compute()) {
    offsetOutput(800.0, 4096.0);
    //      myPID.SetTunings(Kp, Ki, Kd);
    pwmVal0 = (abs(Output) - Output) / 2;
    pwmVal1 = (abs(Output) + Output) / 2;
    analogWrite(pwmPin0, pwmVal0);
    analogWrite(pwmPin1, pwmVal1);
  }

  debouncer2.update();
  switch (currSampleState) {
    case SAMPLE_BACK:
      Setpoint = 0;
      if (debouncer2.fell()) {
        currSampleState = SAMPLE_FORWARD;
        Serial.println("switch to SAMPLE_FORWARD");
      }
      break;
    case SAMPLE_FORWARD:
      Setpoint = -150;
      if (debouncer2.rose()) {
        currSampleState = SAMPLE_BACK;
        Serial.println("switch to SAMPLE_BACK");
      }
      break;
  }
}

void motorOff() {
  analogWrite(pwmPin0, 0);
  analogWrite(pwmPin1, 0);
}

void updateAcc() {
  accX = analogRead(accPinX);
  accY = analogRead(accPinY);
  accZ = analogRead(accPinZ);
}






void printVals() {
  currPrintTime = micros();
    if (currPrintTime - lastPrintTime > printTimeInterval) {
//  if (currPrintTime - lastPrintTime > 5) {
        Serial.print(Input_pos);
        Serial.print(", ");
        Serial.print(thisForce);
        Serial.print(", ");
        Serial.print(accZ-538);
        Serial.print(", ");
    Serial.println(Setpoint);
    //    T_samp = 1000 * ct;
    ct = 0;
    lastPrintTime = currPrintTime;
  }
}
