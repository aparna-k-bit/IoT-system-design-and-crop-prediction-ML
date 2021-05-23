#include "DHT.h"        // including the library of DHT11 temperature and humidity sensor
#include <Arduino.h>
#include <SimpleTimer.h>
#define WLPower 7
#define DHTTYPE DHT11   // DHT 1
#define watersensor A3
#define dht_dpin 5
SimpleTimer timer;
 
float calibration_value = 21.34-1.0;
int phval = 0; 
unsigned long int avgval; 
int buffer_arr[10],temp;
int val = 0;
float ph_act;


String level;

DHT dht(dht_dpin, DHTTYPE); 

void setup() {
  //Define output pins for Mux
  // Set D7 as an OUTPUT
  pinMode(WLPower, OUTPUT);
  // Set to LOW so no power flows through the sensor
  digitalWrite(WLPower, LOW);
  dht.begin();
  Serial.begin(9600); 
 
}

void loop() {
  
  float moisture_percentage;
  int waterlevel;
  
  moisture_percentage = ( 100.00 - ( (analogRead(A1)/1023.00) * 100.00 ) );
  int level = readSensor();
  timer.run(); // Initiates SimpleTimer
 for(int i=0;i<10;i++) 
 { 
 buffer_arr[i]=analogRead(A0);
 delay(30);
 }
 for(int i=0;i<9;i++)
 {
 for(int j=i+1;j<10;j++)
 {
 if(buffer_arr[i]>buffer_arr[j])
 {
 temp=buffer_arr[i];
 buffer_arr[i]=buffer_arr[j];
 buffer_arr[j]=temp;
 }
 }
 }
 avgval=0;
 for(int i=2;i<8;i++)
 avgval+=buffer_arr[i];
 float volt=(float)avgval*5.0/1024/6; 
 ph_act = -5.70 * volt + calibration_value;
 
 Serial.println("pH Val: ");
 delay(1000);
 Serial.print(ph_act);
 Serial.println("\n"); 

 float h = dht.readHumidity();
 float t = dht.readTemperature();         
 Serial.print("Current humidity = ");
 Serial.print(h);
 Serial.print("%  ");
 Serial.print("temperature = ");
 Serial.print(t); 
 Serial.println("C  ");
 Serial.print("Soil Moisture(in Percentage) = ");
 Serial.print(moisture_percentage);
 Serial.println("%");

  
 Serial.print("Water level: ");
 Serial.println(level);
  
  delay(1000);
  
}

int readSensor() {
  digitalWrite(WLPower, HIGH);  // Turn the sensor ON
  delay(10);              // wait 10 milliseconds
  val = analogRead(watersensor);    // Read the analog value form sensor
  digitalWrite(WLPower, LOW);   // Turn the sensor OFF
  return val;             // send current reading
}
