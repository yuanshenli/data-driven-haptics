/* Encoder setup */
#define ENC_A 2 // pin for ENC_A
#define ENC_B 3 // pin for ENC_B
//#define ENC_A 26 // pin for ENC_A
//#define ENC_B 27 // pin for ENC_B
volatile int lastEncoded = 0;
volatile long encoderValue = 0;
volatile long lastEncoderValue = 0;
//int lastMSB = 0;
//int lastLSB = 0;

unsigned long currTime;
unsigned long lastTime = 0;
unsigned long printTime;

void setup() {
  pinMode(ENC_A, INPUT);
  pinMode(ENC_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_A), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), updateEncoder, CHANGE);
  Serial.begin(115200);
}

void loop() {
//  updateEncoder();
  if ((currTime = millis()) - lastTime > printTime) {
    Serial.print(encoderValue);
    Serial.println();
    lastTime = currTime;;
  }
  
//  delay(50);
}


void updateEncoder() {
  int MSB = digitalRead(ENC_A); //MSB = most significant bit
  int LSB = digitalRead(ENC_B); //LSB = least significant bit
  int encoded = (MSB << 1) | LSB; //converting the 2 pin value to single number
  int sum  = (lastEncoded << 2) | encoded; //adding it to the previous encoded value
  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValue ++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValue --;
  lastEncoded = encoded; //store this value for next time
  
 
}
