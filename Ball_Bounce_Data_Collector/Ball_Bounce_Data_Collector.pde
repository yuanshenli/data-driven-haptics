// Ball_Bounce_Data_Collector.pde
/*
Written by: Shenli Yuan
 Date: 07/03/2019
 -------------------------------------------------------------------------------
 */


import processing.serial.*;

Serial myPort;        // The serial port
String [] inData;     // splited incoming data

float pos;
float force;
float acc;

int currState = 0;    // 0: wait,  1: sampling, 2: finish

PrintWriter output;

void setup () {
  size(600, 400); 




  printArray(Serial.list());
  myPort = new Serial(this, Serial.list()[11], 115200);
  myPort.bufferUntil('\n');
}
void draw () {
  switch (currState) {
  case 0: 
    fill(0);
    if (keyPressed) {
      if (key == 's' || key == 'S') {
        currState = 1;   // Start Sampling
        // create data file
        int mo = month();
        int d = day();
        int h = hour();
        int m = minute();
        String fileName = "results/" + str(mo) + "_" + str(d) + "_" + str(h) + "_" + str(m) +".txt";
        output = createWriter(fileName); 
        println(fileName);
      }
    }
    break;
  case 1:
    fill(255);
    if (keyPressed) {
      if (key == 'f' || key == 'F') {
        currState = 2;
        println("finishing");
      }
    }
    break;
  case 2:
    fill(0);
    output.flush();  // Writes the remaining data to the file
    output.close();  // Finishes the file
    currState = 0;
    break;
  }
}

void serialEvent (Serial myPort) {
  // get the ASCII string:
  try {
    if (myPort.available() > 0) {
      String inString = myPort.readStringUntil('\n');
      if (inString != null) {

        inData = split(trim(inString), ',');
        pos = float(inData[0]);
        force = float(inData[1]);
        acc = float(inData[2]);
        String outString = str(pos) + ", " + str(force) + ", " + str(acc);
        if (currState == 1) {
          output.println(outString);
        }  
        //println(outString);
      }
    }
  } 
  catch(RuntimeException e) {
    //e.printStackTrace();
  }
}
