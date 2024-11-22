/*
 SeanContextTwoShockEthoTrig_20210423
 Sean

Starts shock countdown on ethovision trigger.
Gives shock for shockDur.
Hit RESET button to reset for next session.

Wei-li edits 20240604
Huro edits 20241122
 
*/
int ethoPin = 12;
int shockPin = 10;
int ledPin = 13;
int buttonPin = 11;
int fiberPin = 7;

long preShockDelay = 180000; //180000;
//long interShockInt = 58000; // 5000;
long postShockInt = 60000;
int shockDur = 2000;

//LEAVE IT AT 0 even with Ethovision
int hasTrig = 0;

// set start delay
int startDelay = 10;

// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pins
  pinMode(shockPin, OUTPUT);
  //pinMode(ethoPin, INPUT);
  pinMode(fiberPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);
  
  Serial.begin(9600);
  Serial.println("Program starting");
  Serial.println("CFC-A One-Shock");
  Serial.println("Total Time = 242sec");
  Serial.print("Pre-Shock Time = "); Serial.println(preShockDelay);
  Serial.print("Shock Duration = "); Serial.println(shockDur);
  Serial.println("Waiting for ethovision input...");
  Serial.print("Delaying for "); Serial.print(startDelay); Serial.println(" seconds to allow for video recording to start properly");
  // Delay for startDelay
  delay(startDelay * 1000);
}

// the loop function runs over and over again forever
void loop() {
  if ((digitalRead(buttonPin)==LOW) && hasTrig == 0) { //digitalRead(ethoPin)==HIGH || 
    unsigned long startTime = millis();
    Serial.print("button trigger IN, millis=");
    Serial.println(startTime);
    digitalWrite(fiberPin, HIGH);
    while (millis() - startTime < 1000);
    digitalWrite(fiberPin, LOW);

    // Pre-shock countdown
    // while (millis() - startTime < preShockDelay);
    countDown_inSeconds(startTime, preShockDelay, "Shock coming in ");

    // Deliver the shock & get shockTime
    unsigned long shockTime = shockDeliver(1); 
    

    // Wait for duration set by postShockInt
    // while (millis() - shockTime < postShockInt);
    countDown_inSeconds(shockTime, postShockInt, "Session ending in ");

    // end session
    Serial.print("session END, millis=");
    Serial.println(millis());
    digitalWrite(fiberPin, HIGH);
    while (millis() - shockTime < (postShockInt + 500));
    digitalWrite(fiberPin, LOW);
    
    hasTrig = 1;
  }
}

// SUB-FUNCTIONS
unsigned long shockDeliver(int shockNum) {
    digitalWrite(ledPin, HIGH);
    char buffer[20];
    sprintf(buffer, "SHOCK %d OUT, millis=", shockNum);
    Serial.print(buffer);

    unsigned long shockTime = millis();
    Serial.println(shockTime);
    
    digitalWrite(shockPin, HIGH);   // turn the shock on (HIGH is the voltage level)
    digitalWrite(fiberPin, HIGH);
    
    while (millis() - shockTime < shockDur);
    
    digitalWrite(shockPin, LOW); // turn the shock off by making the voltage LOW
    digitalWrite(ledPin, LOW); // wait for the shock duration
    
    sprintf(buffer, "Shock %d OFF, millis=", shockNum);
    Serial.print(buffer);
    Serial.println(millis());
    digitalWrite(fiberPin, LOW);

    return shockTime;
}

void countDown_inSeconds(unsigned long startTime, unsigned long duration, const char* message) {
  while (millis() - startTime < duration) {
    if ((millis() - startTime) % 1000 == 0) {
      Serial.print("\r");
      Serial.print(message);
      char buffer[20];
      sprintf(buffer, "%3d seconds", (duration - (millis() - startTime)) / 1000);
      Serial.print(buffer);
    }
  }
  Serial.println();  // New line after countdown finishes
}