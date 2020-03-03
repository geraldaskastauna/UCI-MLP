package multilayerperceptron;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.Scanner;
import java.util.Arrays;

public class MLP {
	/* Set variables that hold paths for reading files */
	private static final String FIRST_FILE = "src/resources/cw2DataSet1.csv";
	private static final String SECOND_FILE = "src/resources/cw2DataSet2.csv";
	
	/* Weights */
	private static final String HIDDEN_WEIGHTS_FILE = "hWeights.txt";
	private static final String OUTPUT_WEIGHTS_FILE = "oWeights.txt";
	
	/* Variables for array size */
	private static int TOTAL_ELEMENTS = 65;
	private static int TOTAL_ROWS = 2810;
	private static int TOTAL_INPUT_NEURONS = 64;
	private static int TOTAL_HIDDEN_NEURONS = 20;
	private static int TOTAL_OUTPUT_NEURONS = 10;
	
	/* Bias */
	private static int bias = 1;
	
	/* Weights */
	private static double[][] inputToHiddenWeights = new double[TOTAL_INPUT_NEURONS][TOTAL_HIDDEN_NEURONS];
	private static double[][] hiddenToOutputWeights = new double[TOTAL_HIDDEN_NEURONS][TOTAL_OUTPUT_NEURONS];
	
	/* Layers */
	private static double[][] hiddenLayer = new double[TOTAL_ROWS][TOTAL_HIDDEN_NEURONS];
	private static double[][] outputLayer = new double[TOTAL_ROWS][TOTAL_OUTPUT_NEURONS];
	
	/* Arrays with data from files */
	private static double[][] firstFileData = new double[TOTAL_ROWS][TOTAL_ELEMENTS];
	private static double[][] secondFileData = new double[TOTAL_ROWS][TOTAL_ELEMENTS];
	
	/* File read method (returns array) */
	private static double[][] fileRead(String fileName, double[][] array) {
		Scanner scannerObject;
		String inputLine;
		int row = 0;
		
		try {
			scannerObject = new Scanner(new BufferedReader(new FileReader(fileName)));
			
			while(scannerObject.hasNextLine()) {
				inputLine = scannerObject.nextLine();
				
				// Split elements separated by comma from the file
				String[] inputArray = inputLine.split(",");
				
				for(int arrayElement = 0; arrayElement < inputArray.length; arrayElement++) {
					array[row][arrayElement] = Double.parseDouble(inputArray[arrayElement]);
				}
				row++;
			}
		} catch (FileNotFoundException e) {
			System.out.println("File was not found, ERROR: " + e);
		} catch (IndexOutOfBoundsException e) {
			System.out.println("Error reading file, ERROR: " + e);
		}
		return array;
	}
	
	/* Save arrays into file method */
	public static void fileWrite(double[][] array, String filename) {
        PrintStream ps;
	    try {
	    	ps = new PrintStream(new FileOutputStream(filename));
	        for(int row=0;row < array.length;row++){
	        	if(row != 0)
	        		ps.println();
	           for(int col=0; col < array[row].length;col++){
	                    double s = array[row][col];
	                    ps.print(s + ",");
	                }
	            }
	            ps.close();
	        } catch (FileNotFoundException e) {
	            System.out.println(e.getMessage());
	        }
	}
	
	/* Store both files data into arrays and fill weights array */
	public static void storeData() {
		fileRead(FIRST_FILE, firstFileData);
		fileRead(SECOND_FILE, secondFileData);
		fileRead(HIDDEN_WEIGHTS_FILE, inputToHiddenWeights);
		fileRead(OUTPUT_WEIGHTS_FILE, hiddenToOutputWeights);
	}
	
	/* Sigmoid derivative */
	private static double dersigmoid(double value) {
		return (value * (1 - value));
	}
	
	/* Sigmoid function (Vanishing gradient problem) */
    private static double sigmoid(double value) {
        return (1 / (1 + Math.exp(-value)));
    }
    
	/* Get dot product */
    public static double[][] dotProduct(double[][] input, double[][] hidden, double[][] weights){
        double neuronValue = 0;
        
        // Loop through data file
        for(int array = 0; array < input.length; array++) {
	        // Loop through hidden layer neurons
	        for(int secondLayerNeuron = 0; secondLayerNeuron < hidden[array].length; secondLayerNeuron++) {
	            // Loop through input layer neurons and reset neuronValue before each loop
	            neuronValue = 0;
	                for(int firstLayerNeuron = 0; firstLayerNeuron < input[array].length - 1; firstLayerNeuron++){
	                    neuronValue += input[array][firstLayerNeuron] * weights[firstLayerNeuron][secondLayerNeuron];
	                }
	                // Store neuronValue + bias after activation function into array
	                hidden[array][secondLayerNeuron] = sigmoid(neuronValue + bias);;
	        }
        }
	    return hidden;
    }
    
    /* Train network */
    private static void trainNetwork(double[][] input, double teachingStep, double lmse, double momentum, int maxEpochs) {
    	double mse = 0.0;
    	int epochs = 1;
    	double error = 0.0;
    	double target;
    	
    	// Hidden layer delta
    	double[][] hdelta = new double[TOTAL_ROWS][TOTAL_HIDDEN_NEURONS];
    	
    	// Output layer delta
    	double[][] odelta = new double[TOTAL_ROWS][TOTAL_OUTPUT_NEURONS];

    	// Temporary weights for training
    	double[][] tempHiddenWeights = Arrays.copyOf(inputToHiddenWeights, inputToHiddenWeights.length);
    	double[][] tempOutputWeights = Arrays.copyOf(hiddenToOutputWeights, hiddenToOutputWeights.length);

    	// Previous weights for training
       	double[][] previousHiddenWeights = Arrays.copyOf(inputToHiddenWeights, inputToHiddenWeights.length);
    	double[][] previousOutputWeights = Arrays.copyOf(hiddenToOutputWeights, hiddenToOutputWeights.length);

        // Loop until error is small enough
        while(Math.abs(mse - lmse) > 0.0001) {
            //for each epoch reset the mean square error
            mse = 0.0;
            
            //for each array from file
            for(int fileArray = 0; fileArray < input.length; fileArray++) {
            
	            // Calculate input to hidden layer
	        	dotProduct(input, hiddenLayer, inputToHiddenWeights);
			
	        	// Calculate hidden to output layer
	        	dotProduct(hiddenLayer, outputLayer, hiddenToOutputWeights);
            	
        		// Set the target which is the last element of input array
            	target = input[fileArray][64];
  
                //Backpropagation from output layer
                for(int outputNeuron = 0; outputNeuron < outputLayer[fileArray].length; outputNeuron++) {
                    //Calculate delta and error
                    if(outputNeuron != target) {
                        odelta[fileArray][outputNeuron] = (0.0 - outputLayer[fileArray][outputNeuron]) * dersigmoid(outputLayer[fileArray][outputNeuron]);
                        error += (0.0 - outputLayer[fileArray][outputNeuron]) * (0.0 - outputLayer[fileArray][outputNeuron]);
                    } else {
                        odelta[fileArray][outputNeuron] = (1.0 - outputLayer[fileArray][outputNeuron]) * dersigmoid(outputLayer[fileArray][outputNeuron]);
                        error += (1.0 - outputLayer[fileArray][outputNeuron]) * (1.0 - outputLayer[fileArray][outputNeuron]);
                    }
                }

                // Backpropagation from hidden layer
                for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer[fileArray].length; hiddenNeuron++) {
                	//zero the values from the previous iteration
                    hdelta[fileArray][hiddenNeuron] = 0.0; 

                    //Add to the delta for each output neuron
                    for(int outputNeuron = 0; outputNeuron < outputLayer[fileArray].length; outputNeuron++) {
                        hdelta[fileArray][outputNeuron] += odelta[fileArray][outputNeuron] * inputToHiddenWeights[hiddenNeuron][outputNeuron] ;
                    }

                    // Derivative for later weight adjustments
                    hdelta[fileArray][hiddenNeuron] *= dersigmoid(hiddenLayer[fileArray][hiddenNeuron]);
                }

                // Weights modification
                tempHiddenWeights = Arrays.copyOf(inputToHiddenWeights, inputToHiddenWeights.length);
            	tempOutputWeights = Arrays.copyOf(hiddenToOutputWeights, hiddenToOutputWeights.length);

                // Input to hidden weights
                for(int inputNeuron = 0; inputNeuron < input[fileArray].length - 1; inputNeuron++) {
                    for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer[fileArray].length; hiddenNeuron++) {
                        inputToHiddenWeights[inputNeuron][hiddenNeuron] +=
                        					(momentum * (inputToHiddenWeights[inputNeuron][hiddenNeuron] 
                        					- previousHiddenWeights[inputNeuron][hiddenNeuron]))
                        					+ (teachingStep * hdelta[fileArray][hiddenNeuron] * input[fileArray][inputNeuron]);
                    }
                }

                // Hidden to output weights
                for(int outputNeuron = 0; outputNeuron < outputLayer[fileArray].length; outputNeuron++) {
                    for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer[fileArray].length; hiddenNeuron++) {
                    	hiddenToOutputWeights[hiddenNeuron][outputNeuron] +=
                                			(momentum * (hiddenToOutputWeights[hiddenNeuron][outputNeuron]
                                			- previousOutputWeights[hiddenNeuron][outputNeuron]))
                                			+ (teachingStep * odelta[fileArray][outputNeuron] * hiddenLayer[fileArray][hiddenNeuron]);
                    }
                }
                
                // Save modified weights as previous for each loop
                previousHiddenWeights = Arrays.copyOf(tempHiddenWeights, tempHiddenWeights.length);
            	previousOutputWeights = Arrays.copyOf(tempOutputWeights, tempOutputWeights.length);
                
                // Get total mean squared error for epoch
                mse += error / (TOTAL_OUTPUT_NEURONS + 1);
                
                // Reset error for next loop
                error = 0.0;
            }
            
            // Print the process
            System.out.println("Epoch: " + epochs + " | mean square error value = " + mse);
            
            if(epochs == maxEpochs) 
           		break;
            
            epochs++;
            
            fileWrite(previousHiddenWeights, "newWeightsHidden.txt");
            fileWrite(previousOutputWeights, "newWeightsOutput.txt");
        }
    } 
    
    /* Round double values to one decimal point */
	public static double roundToTwoDecimals(double value) {
									// value = 123.123
		value = value * 100; 		// value = 12312.3
		value = Math.round(value); 	// value = 12312
		value = value / 100; 		// value = 123.12
		
		return value;
	}
    
    /* Get accuracy */
    public static void getAccuracy(double[][] data) {
        double winner = 0;
        int indexJ = 0;
        double correctPredictionCounter = 0;
        double accuracy = 0;
        
        // Loop through the output layer
        for(int i = 0; i < outputLayer.length; i++){
            // Reset values for each loop
            indexJ = 0;
            winner = 0;
            // Loop through the output elements and check for the winner
            for(int j = 0; j < outputLayer[i].length; j++){
                if(outputLayer[i][j] > winner){
                    winner = outputLayer[i][j];
                    indexJ = j;
                }
            }
            // Count correct predictions
            if(indexJ == data[i][64]) 
            	correctPredictionCounter++;
        }
        // Count the accuracy
        accuracy = (correctPredictionCounter / data.length) * 100;
        
        // Print the results
        System.out.println("Total inputs: " + data.length);
        System.out.println("Correct predicitons: " + (int) correctPredictionCounter);
        System.out.println("Accuracy percentage: " + roundToTwoDecimals(accuracy) + "%");   
    }
    
	/* Initialise program */
	public static void initialise(double[][] input, double teachingStep, double lmse, double momentum, int maxEpochs) {
		// Store data
		storeData();
		trainNetwork(input, teachingStep, lmse, momentum, maxEpochs);
		getAccuracy(input);
	}
	
	public static void main(String[] args) {
		// Initialise MLP
		/* PARAMETERS double[][] inputArray, double teachingStep, double lmse, double momentum, int maxEpochs */
		initialise(firstFileData, 0.04, 0.01, 0.5, 1000);
	}
}
