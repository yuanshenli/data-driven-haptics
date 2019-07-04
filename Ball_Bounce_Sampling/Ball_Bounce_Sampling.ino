
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
float A_samp = 150.0;
int T_samp = 1000;
int t_samp = 0;
unsigned int t_ms_curr;
unsigned int t_ms_last = 0;
int t_res = 1;  // [ms]


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
  T_samp = analogRead(34);
  t_ms_curr = millis();
  if (t_ms_curr - t_ms_last > t_res) {
    t_ms_last = t_ms_curr;
    Setpoint = A_samp * sin(2.0 * M_PI / (float)T_samp * (float)t_samp) - A_samp;
    t_samp++;
    t_samp %= T_samp;
  }
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
  currPrintTime = millis();
  if (currPrintTime - lastPrintTime > printTimeInterval) {
    Serial.print(Input_pos);
    Serial.print(", ");
    Serial.print(thisForce);
    Serial.print(", ");
    Serial.print(accZ-538);
    Serial.print(", ");
    Serial.println(ct);
    ct = 0;
    lastPrintTime = currPrintTime;
  }
}
