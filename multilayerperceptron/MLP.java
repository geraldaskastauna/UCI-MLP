package multilayerperceptron;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.Scanner;
import java.util.Arrays;

public class MLP {
	/* Input files path constants*/
	private static final String FIRST_FILE = "src/resources/cw2DataSet1.csv";
	private static final String SECOND_FILE = "src/resources/cw2DataSet2.csv";
	
	/* Weights files constants*/
	private static final String HIDDEN_WEIGHTS_FILE = "src/resources/hiddenWeights.txt";
	private static final String OUTPUT_WEIGHTS_FILE = "src/resources/outputWeights.txt";
	
	/* Constants */
	private static final int TOTAL_ELEMENTS = 65;
	private static final int TOTAL_ROWS = 2810;
	private static final int TOTAL_INPUT_NEURONS = 64;
	private static final int TOTAL_HIDDEN_NEURONS = 20;
	private static final int TOTAL_OUTPUT_NEURONS = 10;
	
	/* Bias constant*/
	private static final int BIAS = 1;
	
	/* Weights */
	private static double[][] inputToHiddenWeights = new double[TOTAL_INPUT_NEURONS][TOTAL_HIDDEN_NEURONS];
	private static double[][] hiddenToOutputWeights = new double[TOTAL_HIDDEN_NEURONS][TOTAL_OUTPUT_NEURONS];
	
	/* Layers */
	private static double[][] hiddenLayer = new double[TOTAL_ROWS][TOTAL_HIDDEN_NEURONS];
	private static double[][] outputLayer = new double[TOTAL_ROWS][TOTAL_OUTPUT_NEURONS];
	
	/* Arrays for data from files */
	private static double[][] firstFileData = new double[TOTAL_ROWS][TOTAL_ELEMENTS];
	private static double[][] secondFileData = new double[TOTAL_ROWS][TOTAL_ELEMENTS];
	
	/* Method that reads in file data from CSV (comma separated values) files.
	 * Requires 2 parameters:
	 * 1st parameter is String type named fileName: 
	 * Example: "example.csv" or use constants "FIRST_FILE".
	 * 2nd parameter is double type 2D array named fileDataArray:
	 * An array that holds the data from the file.
	 * Method returns type double 2D array.
	 */
	private static double[][] fileRead(String fileName, double[][] fileDataArray) {
		Scanner scannerObject;
		String inputLine;
		int row = 0;
		
		try {
			scannerObject = new Scanner(new BufferedReader(new FileReader(fileName)));
			
			while(scannerObject.hasNextLine()) {
				inputLine = scannerObject.nextLine();
				String[] delimiter = inputLine.split(",");
				
				// Split elements separated by comma from the file
				String[] inputArray = delimiter;
				
				for(int arrayElement = 0; arrayElement < inputArray.length; arrayElement++) {
					// Save as double values
					fileDataArray[row][arrayElement] = Double.parseDouble(inputArray[arrayElement]);
				}
				// Count total rows
				row++;
			}
		// Error catch clauses
		} catch (FileNotFoundException errorName) {
			System.out.println("File was not found, ERROR: " + errorName);
		} catch (IndexOutOfBoundsException errorName) {
			System.out.println("Error reading file, ERROR: " + errorName);
		}
		return fileDataArray;
	}
	
	/* Void type method that saves an array data into a file.
	 * Requires 2 parameters:
	 * 1st parameter is String type named fileName in which it saves the data (file will be created if it does not exist): 
	 * Example: "example.txt"
	 * 2nd parameter is double type 2D array named data:
	 * An array that holds the data that will be written to file.
	 */
	public static void fileWrite(String fileName, double[][] data) {
		// Create PrintStream class object
        PrintStream printStreamObj;
        double arrayElement;
        String path = "src/resources/";
        
	    try {
	    	printStreamObj = new PrintStream(new FileOutputStream(path + fileName));
	    	
	        for(int row = 0; row < data.length; row++){
	        	// Dont make a new line on the first iteration
	        	if(row != 0)
	        		// Next line
	        		printStreamObj.println();
	           for(int column = 0; column < data[row].length; column++){
	                    arrayElement = data[row][column];
	                    // Separate by comma;
	                    printStreamObj.print(arrayElement + ",");
	                }
	            }
	        // Close the writing into file
	        printStreamObj.close();
	        } catch (FileNotFoundException errorName) {
	            System.out.println("File was not created. ERROR: " + errorName.getMessage());
	        }
	}
	
	/* Void type method that reads data from files using previous fileRead method.
	 * Stores data from FIRST_FILE constant to firstFileData array.
	 * Stores data from SECOND_FILE constant to secondFileData array.
	 * Stores data from HIDDEN_WEIGHTS_FILE constant to inputToHiddenWeights array.
	 * Stores data from HIDDEN_WEIGHTS_FILE constant to hiddenToOutputWeights array.
	 */
	public static void storeData() {
		fileRead(FIRST_FILE, firstFileData);
		fileRead(SECOND_FILE, secondFileData);
		fileRead(HIDDEN_WEIGHTS_FILE, inputToHiddenWeights);
		fileRead(OUTPUT_WEIGHTS_FILE, hiddenToOutputWeights);
	}
	
	/* Sigmoid derivative formula.
	 * Method returns type double value.
	 */
	private static double sigmoidDerivative(double value) {
		return (value * (1 - value));
	}
	
	/* Sigmoid function formula.
	 * Method returns type double value.
	 */
    private static double sigmoid(double value) {
        return (1 / (1 + Math.exp(-value)));
    }
    
	/* Method that calculates the dot product between two layers using weights between them.
	 * Requires 3 parameters:
	 * 1st parameter is double type 2D array named startingLayer:
	 * Its an input layer or hidden layer if going from hidden to output (or to next hidden).
	 * 2nd parameter is double type 2D array named nextLayer:
	 * Its a layer with all 0 values that is connected with our startingLayer.
	 * 3rd parameter is double type 2D array named weights:
	 * Its an array that holds weights values between startingLayer and nextLayer.
	 * Method returns nextLayer double type 2D array with filled values
	 */
    public static double[][] dotProduct(double[][] startingLayer, double[][] nextLayer, double[][] weights){
    	double neuronValue = 0;
        
        // Loop through total starting layer neurons
        for(int startingNeuron = 0; startingNeuron < startingLayer.length; startingNeuron++) {
	        // Loop through next layer neurons
	        for(int nextLayerNeuron = 0; nextLayerNeuron < nextLayer[startingNeuron].length; nextLayerNeuron++) {
	            // Loop through input layer neurons and reset neuronValue before each loop
	            neuronValue = 0;
	                for(int startingLayerNeuron = 0; startingLayerNeuron < startingLayer[startingNeuron].length - 1; startingLayerNeuron++){
	                	// Count total neuronValue for one next layer neuron
	                    neuronValue += startingLayer[startingNeuron][startingLayerNeuron] * weights[startingLayerNeuron][nextLayerNeuron];
	                }
	                // Store neuronValue + bias after activation function into array
	                nextLayer[startingNeuron][nextLayerNeuron] = sigmoid(neuronValue + BIAS);;
	        }
        }
	    return nextLayer;
    }
    
    /* Void type method that gets dot products for MLP layers
     * Requires 1 parameter:
     * 1st parameter is double type 2D array named inputLayer:
     * Its an input layer.
     * Example: firstFileData or secondFileData
     */
    public static void calculateDotProducts(double[][] inputLayer) {
    	// Calculate dot product from input to hidden layer
    	dotProduct(inputLayer, hiddenLayer, inputToHiddenWeights);
	
    	// Calculate dot product from hidden to output layer
    	dotProduct(hiddenLayer, outputLayer, hiddenToOutputWeights);
    }
    
    /* Method that trains the neural network that has one hidden layer using backpropagation method.
	 * Requires 5 parameters:
	 * 1st parameter is double type 2D input array:
	 * Its an array that hold the inputs (an input layer).
	 * Examples: firstFileData, secondFileData
	 * 2nd parameter is double type value called learningRate:
	 * Its a value used for MLP training. Higher value of this could mean faster learning, but also bigger vanishing gradient
	 * which would lead to less accurate results.
	 * Examples: Values between 0.01 and 0.1. I chose something like 0.04.
	 * 3rd parameter is a double type value called lmse (least mean squared error):
	 * Its a goal value for all neurons that are not our target neuron.
	 * Example: Usual value is 0.01.
	 * 4th parameter is a double type value called momentum
	 * Its a value used for MLP training to optimise gradient descent.
	 * Example: Value between 0 and 1. I chose something like 0.4.
	 * 5th parameter is int type value called maxEpochs
	 * Its a value that sets the maximum amount of epochs for training.
	 * Example: I did 300 epochs and got ~63% accuracy. 1000 epochs could be good for accuracy, but it takes time to train.
	 */
    private static void trainMLP(double[][] input, double learningRate, double lmse, double momentum, int maxEpochs) {
    	// Mean squared error variable
    	double mse = 0.0;
    	
    	// Epochs counter variable
    	int countEpochs = 1;
    	
    	// Total error variable
    	double error = 0.0;
    	
    	// Lowest error threshold variable
    	double errorThreshold = 0.0001;
    	
    	// Target variable (last element of input array)
    	double target;
    	
    	// Target position in array
    	int targetPosition = TOTAL_ELEMENTS - 1;
    	
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

        // Loop until errorThreshold is reached
        while(Math.abs(mse - lmse) > errorThreshold) {
            // For each epoch reset the mean square error
            mse = 0.0;
        	
            // Loop through all the inputs
            for(int fileArray = 0; fileArray < input.length; fileArray++) {
            
            	// Calculate dot products from input to hidden and from hidden to output
            	calculateDotProducts(input);
            	
        		// Set the target which is the last element of input array
            	target = input[fileArray][targetPosition];
  
                // Backpropagation from output layer
                for(int outputNeuron = 0; outputNeuron < outputLayer[fileArray].length; outputNeuron++) {
                    // Calculate delta and error if output neuron IS NOT the target
                    if(outputNeuron != target) {
                        odelta[fileArray][outputNeuron] = (0.0 - outputLayer[fileArray][outputNeuron]) * sigmoidDerivative(outputLayer[fileArray][outputNeuron]);
                        error += (0.0 - outputLayer[fileArray][outputNeuron]) * (0.0 - outputLayer[fileArray][outputNeuron]);
                    } 
                    // Calculate delta and error if output neuron IS the target
                    else {
                        odelta[fileArray][outputNeuron] = (1.0 - outputLayer[fileArray][outputNeuron]) * sigmoidDerivative(outputLayer[fileArray][outputNeuron]);
                        error += (1.0 - outputLayer[fileArray][outputNeuron]) * (1.0 - outputLayer[fileArray][outputNeuron]);
                    }
                }

                /* Backpropagation from hidden layer */
                for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer[fileArray].length; hiddenNeuron++) {
                	// Zero the values from the previous iteration
                    hdelta[fileArray][hiddenNeuron] = 0.0; 

                    // Add to the delta for each output neuron
                    for(int outputNeuron = 0; outputNeuron < outputLayer[fileArray].length; outputNeuron++) {
                        hdelta[fileArray][outputNeuron] += odelta[fileArray][outputNeuron] * inputToHiddenWeights[hiddenNeuron][outputNeuron] ;
                    }

                    // Use sigmoid derivative for later weight adjustments
                    hdelta[fileArray][hiddenNeuron] *= sigmoidDerivative(hiddenLayer[fileArray][hiddenNeuron]);
                }

                // Weights modification
                tempHiddenWeights = Arrays.copyOf(inputToHiddenWeights, inputToHiddenWeights.length);
            	tempOutputWeights = Arrays.copyOf(hiddenToOutputWeights, hiddenToOutputWeights.length);

                /* Input to hidden weights */
                for(int inputNeuron = 0; inputNeuron < input[fileArray].length - 1; inputNeuron++) {
                    for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer[fileArray].length; hiddenNeuron++) {
                        inputToHiddenWeights[inputNeuron][hiddenNeuron] +=
                        					(momentum * (inputToHiddenWeights[inputNeuron][hiddenNeuron] 
                        					- previousHiddenWeights[inputNeuron][hiddenNeuron]))
                        					+ (learningRate * hdelta[fileArray][hiddenNeuron] * input[fileArray][inputNeuron]);
                    }
                }

                /* Hidden to output weights */
                for(int outputNeuron = 0; outputNeuron < outputLayer[fileArray].length; outputNeuron++) {
                    for(int hiddenNeuron = 0; hiddenNeuron < hiddenLayer[fileArray].length; hiddenNeuron++) {
                    	hiddenToOutputWeights[hiddenNeuron][outputNeuron] +=
                                			(momentum * (hiddenToOutputWeights[hiddenNeuron][outputNeuron]
                                			- previousOutputWeights[hiddenNeuron][outputNeuron]))
                                			+ (learningRate * odelta[fileArray][outputNeuron] * hiddenLayer[fileArray][hiddenNeuron]);
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
            System.out.println("Epoch: " + countEpochs + " | Error value = " + mse);
            
            // Check for the epoch count
            if(countEpochs == maxEpochs) 
           		break;
            
            // Add epochs
            countEpochs++;
            
            // Save weights into a file after each epoch
            fileWrite("newWeightsHidden.txt", previousHiddenWeights);
            fileWrite("newWeightsHidden.txt", previousOutputWeights);
        }
    } 
    
    /* Method that rounds double values to two decimal point.
     * Requires 1 parameter:
     * 1st parameter is any double type value.
     * Working example is commented in the method.
     * Method returns double type value
     */
	public static double roundToTwoDecimals(double value) {
									// value = 123.123
		value = value * 100; 		// value = 12312.3
		value = Math.round(value); 	// value = 12312
		value = value / 100; 		// value = 123.12
		
		return value;
	}
    
    /* Void type method that calculates the accuracy for chosen 2D input array
     * Requires 1 parameter:
     * 1st parameter is double type 2D array named data:
     * Its an array that has input data. 
     * Example: FIRST_FILE or SECOND_FILE constants
     * Method prints the results to the console
     */
    public static void getAccuracy(double[][] data) {
    	double winner = 0;
        int correctAnswerPos = 0;
        double correctPredictionCounter = 0;
        double accuracy = 0;
        
        // Loop through the output layer
        for(int outputNeuron = 0; outputNeuron < outputLayer.length; outputNeuron++){
            // Reset values for each loop
        	correctAnswerPos = 0;
            winner = 0;
            // Loop through the output elements and check for the winner
            for(int layerElement = 0; layerElement < outputLayer[outputNeuron].length; layerElement++){
            	// Check for the winner and save its position
                if(outputLayer[outputNeuron][layerElement] > winner){
                    winner = outputLayer[outputNeuron][layerElement];
                    correctAnswerPos = layerElement;
                }
            }
            // Count correct predictions
            if(correctAnswerPos == data[outputNeuron][64]) 
            	correctPredictionCounter++;
        }
        // Count the accuracy
        accuracy = (correctPredictionCounter / data.length) * 100;
        
        // Print the results
        System.out.println("Total inputs: " + data.length);
        System.out.println("Correct predicitons: " + (int) correctPredictionCounter);
        System.out.println("Accuracy percentage: " + roundToTwoDecimals(accuracy) + "%");   
    }
    
    /* Void type method to print out the results */
    public static void printResults() {
    	System.out.println("First file data: \n");
    	calculateDotProducts(firstFileData);
		getAccuracy(firstFileData);
		
		System.out.println("______________________________________\n");
		System.out.println("Second file data:\n");
		calculateDotProducts(secondFileData);
		getAccuracy(secondFileData);
		
		System.out.println("\n\nTraining was done on first file for ~300 epochs");
    }
    
	/* Void type method that  starts the whole program. */
	public static void initialise(double[][] input, double learningRate, double lmse, double momentum, int maxEpochs) {
		storeData();
		
		/* Uncomment this for training process */
		//trainMLP(input, learningRate, lmse, momentum, maxEpochs);
		//getAccuracy(input);
		
		
		/* Comment this when training */
		printResults();
	}
	
	public static void main(String[] args) {
		/* parameters are commented above trainMLP method */
		initialise(firstFileData, 0.04, 0.01, 0.4, 1000);
	}
}
