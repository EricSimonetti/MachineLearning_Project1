import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Main {
    //program parameters
    final static boolean DEBUG_MODE = false;
    final static double LEARNING_RATE = .001,
                        LASSO_LAMBDA_START = 196, //196 .28
                        LASSO_LAMBDA_END = 196,   //set equal to start for no testing
                        DELTA_LAMBDA = 1; //cant be zero!!!
    final static int ITERATIONS = 200;
    public static void main(String[] args) {
        final boolean CALCULATE_DEPENDENCIES = true,
            ASSUME_STRINGS_CATEGORIGAL = true;

        //data information
        final ArrayList<String> excludedData = new ArrayList<String>(
                Arrays.asList("Id")),
            otherCategoricalData = new ArrayList<String>(
                Arrays.asList("MSSubClass",
                              "OverallQual",
                              "OverallCond"));
        final String DEPENDENT_VARIABLE = "SalePrice",
            RAW_DIRECTORY = "Data/Raw/",
            PROCESSED_DIRECTORY = "Data/Processed/",
            OUTPUT_DIRECTORY = "Data/Output/output.csv";

        CSVHandler csvh = new CSVHandler(DEBUG_MODE);
        ArrayList<ArrayList<Double>> processedData;
        if(CALCULATE_DEPENDENCIES) {
            processedData = csvh.processCSV(RAW_DIRECTORY + "train.csv",
                    PROCESSED_DIRECTORY,
                    DEPENDENT_VARIABLE,
                    ASSUME_STRINGS_CATEGORIGAL,
                    otherCategoricalData,
                    excludedData);
        }
        else{
            processedData = csvh.readProcessedCSV(PROCESSED_DIRECTORY);
        }
        if(DEBUG_MODE) System.out.println("Processed Data successfully loaded");

        Random random = new Random();
        RealVector weights = new ArrayRealVector(random.doubles(processedData.size()-1).toArray());
        double bias = random.nextDouble();

        RealVector Y = new ArrayRealVector(processedData.remove(processedData.size()-1)
                .stream()
                .mapToDouble(Double::doubleValue)
                .toArray()
        );

        double[][] processedDataArray = new double[processedData.size()][];
        for (int i = 0; i<processedData.size(); i++) {
            processedDataArray[i] = processedData.get(i)
                    .stream()
                    .mapToDouble(Double::valueOf)
                    .toArray();
        }
        RealMatrix X = new Array2DRowRealMatrix(processedDataArray).transpose();
        //LASSO
        double meanPercentageError = 0;

        for(double lasso = LASSO_LAMBDA_START; lasso <= LASSO_LAMBDA_END; lasso+=DELTA_LAMBDA) {
            for (int j = 0; j < ITERATIONS; j++) {
                RealVector predictedY = X.operate(weights).mapAdd(bias);
                meanPercentageError = meanAbsolutePercentageError(Y, predictedY);
                //System.out.println(meanPercentageError);
                weights = updateWeights(X, Y, predictedY, weights, lasso);
                bias = updateBias(Y, predictedY, bias);
                if(meanPercentageError>100) j = ITERATIONS;
            }
            System.out.println("MAPE: " + meanPercentageError + " Lasso: " + lasso);
        }

        //    <-----------------------------TESTING--------------------------------->
        ArrayList<ArrayList<Double>> testData = csvh.getTestingData(RAW_DIRECTORY + "test.csv",
                PROCESSED_DIRECTORY,
                DEPENDENT_VARIABLE);

        ArrayList<Double> ids = testData.remove(0);
        //testData.trimToSize();
        processedDataArray = new double[testData.size()][];


        for (int i = 0; i<testData.size(); i++) {
            processedDataArray[i] = testData.get(i)
                    .stream()
                    .mapToDouble(Double::valueOf)
                    .toArray();
        }
        RealMatrix testX = new Array2DRowRealMatrix(processedDataArray);
        testX = testX.transpose();
        double[] result = testX.operate(weights).mapAdd(bias).toArray();

        String[][] output = new String[2][];
        output[0] = new String[result.length+1];
        output[0][0] = "SalePrice";
        output[1] = new String[result.length+1];
        output[1][0] = "Id";

        for(int i = 0; i<result.length; i++){
            output[0][i+1] = (result[i]+"");
            output[1][i+1] = (ids.get(i).toString());
        }

        System.out.println("\n\nRESULTS:");
        for(String[] arr : output) {
            System.out.printf("%-12s",arr[0] + ": ");
            for (int j = 1; j<arr.length; j++) {
                System.out.printf("%-30s",arr[j] + " ");
            }
            System.out.println();
        }
        System.out.println("Final mean percentage error: " + meanPercentageError);
        csvh.writeOutputCSV(OUTPUT_DIRECTORY, output);
    }

    // <-------------------------- LASSO implementation functions ------------------------------>
    private static RealVector updateWeights(RealMatrix X, RealVector Y, RealVector predictedY, RealVector weights, double lasso){
        int numDataPoints = weights.getDimension();
        RealVector updatedWeights = new ArrayRealVector(new double[numDataPoints]);
        RealVector L1 = proximalL1Norm(weights, LEARNING_RATE);
        double dW;
        for(int i = 0; i<numDataPoints; i++){
            double difference = Y.getEntry(i) - predictedY.getEntry(i);
            double sumOfDotProducts = Arrays.stream((X.getColumnVector(i)).mapMultiply(difference).toArray()).sum();
            //this is the result of summing all dot products of the columns of X and Y-predictedY

            dW = (-2*(sumOfDotProducts)/numDataPoints + lasso*L1.getEntry(i));

            updatedWeights.setEntry(i, weights.getEntry(i) - LEARNING_RATE * dW);
        }
        return updatedWeights;
    }

    private static double updateBias(RealVector Y, RealVector predictedY, double bias){
        double scalingFactor = -2/Y.getDimension();
        double dB = scalingFactor * Arrays.stream(Y.subtract(predictedY).toArray()).sum();
        return bias - LEARNING_RATE * dB;
    }

    private static RealVector proximalL1Norm(RealVector vector, double alpha){
        RealVector v = new ArrayRealVector(vector.toArray());
        for(int i = 0; i<v.getDimension(); i++){
            if(v.getEntry(i) >= alpha)
                v.addToEntry(i, -alpha);
            else if(v.getEntry(i) <= -alpha)
                v.addToEntry(i, alpha);
            else{
                v.setEntry(i, 0);
            }
        }
        return v;
    }

    private static double rootMeanSquaredError(RealVector Y, RealVector predictedY){
        RealVector residual = Y.subtract(predictedY);
        return Math.sqrt(Arrays.stream(residual.ebeMultiply(residual).toArray()).sum());
    }

    private static double meanAbsolutePercentageError(RealVector Y, RealVector predictedY){
        double numberOfDataPoints = Y.getDimension();
        RealVector percentageError = Y.subtract(predictedY).ebeDivide(Y).map(Math::abs);
        return (1/numberOfDataPoints) * Arrays.stream(percentageError.toArray()).sum();
    }
}
