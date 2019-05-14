
#include <DDHapticHelper.h>
float myKp = 75;
float myKi = 0.75;
float myKd = 250; //  * Kp;

bool isSaturated = false;
float lastPos = 0;
float lastForce = 0;
float maxForce = 620;
float forceIncFill = 20;
float posIncFill = posRes;


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
  for (int i = 0; i < filterWindowSize; i++) {
    filterWindow[i] = 0;
  }
}


void loop() {
  if (!isSaturated) {
    toggleState();
    updateEncoderAB();
    filterEncoderAB();
    Input = Input_pos;
    int thisForce = updateRawForce();
  
    if (myPID.Compute()) {
      offsetOutput(800.0, 4096.0);  
      myPID.SetTunings(Kp, Ki, Kd);
      pwmVal0 = (abs(Output) - Output) / 2;
      pwmVal1 = (abs(Output) + Output) / 2;
      analogWrite(pwmPin0, pwmVal0);
      analogWrite(pwmPin1, pwmVal1);
    }
    // Take sample and take into the buffers
    posBuffer[bufCount] = Input;
    forceBuffer[bufCount] = thisForce;
  } else {
    posBuffer[bufCount] = lastPos + posIncFill;
    forceBuffer[bufCount] = lastForce + forceIncFill;
    analogWrite(pwmPin0, 0);
    analogWrite(pwmPin1, 0);
  }
  bufCount++;
  // When buffers are full, take the average of buffer and save into data arrays
  if (bufCount == bufSize) {
    posData[dataCount] = averageBuf(posBuffer, bufSize);
    forceData[dataCount] = averageBuf(forceBuffer, bufSize);
    if (forceData[dataCount] >= maxForce && lastForce >= maxForce) {
      isSaturated = true;
    }
    lastPos = posData[dataCount];
    lastForce = forceData[dataCount];
    dataCount++;
    Setpoint += posRes;
    bufCount %= bufSize;
    
    
  }
  // When data arrays are full, print them
  if (dataCount == dataSize) {
    Serial.print("pos = ");
    printBuf(posData, dataSize);
    Serial.print("force = ");
    printBuf(forceData, dataSize);
    analogWrite(pwmPin0, 0);
    analogWrite(pwmPin1, 0);
    while(true);
  }
  
  printVals();
}

void printVals() {
    currPrintTime = millis();
    if (currPrintTime - lastPrintTime > printTimeInterval) {
      Serial.print(Setpoint);
      Serial.print(", ");
      Serial.print(Input);
      Serial.print(", ");
//      Serial.print(dataCount);
//      Serial.print(", ");
      Serial.print(forceBuffer[bufCount]);
      Serial.println();
      lastPrintTime = currPrintTime;
    }
}
