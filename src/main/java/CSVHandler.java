import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.stream.Collectors;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

class CSVHandler {
    //program parameters
    private final double CORR_COEFF_CUTOFF = .6;
    private final boolean REMOVE_OUTLIERS = false;
    private boolean debugMode;

    CSVHandler(boolean debugMode){ this.debugMode = debugMode; }
    CSVHandler() {this.debugMode = false;}


    ArrayList<ArrayList<Double>> processCSV(String rawDirectory, String processedDirectory, String dependentVariable, boolean assumeStringsCategorical, ArrayList<String> otherCategoricalData, ArrayList<String> excludedData){
        String[][] raw = readCSV(rawDirectory, true);
        ArrayList<ArrayList<String>> normalizationFactors = new ArrayList<>();

        System.out.println("Calculating Normalization factors... ");
        calculateNormalizationFactors(normalizationFactors,
                raw,
                dependentVariable,
                assumeStringsCategorical,
                otherCategoricalData,
                excludedData
        );
        System.out.println("Processing Data... ");
        ArrayList<ArrayList<Double>> processedData = processData(true, normalizationFactors, raw, dependentVariable);

        ArrayList<String[]> toWrite = normalizationFactors.stream()
                .map(x -> x.toArray(new String[0]))
                .collect(Collectors.toCollection(ArrayList::new));

        writeCSV(processedDirectory + "normalizationFactors.csv", toWrite);

        toWrite.clear();
        for(ArrayList<Double> feild : processedData) {
            String[] s = new String[feild.size()];
            for (int j = 0; j<s.length; j++) {
                s[j] = feild.get(j).toString();
            }
            toWrite.add(s);
        }

        writeCSV(processedDirectory + "processedData.csv", toWrite);
        return processedData;
    }

    ArrayList<ArrayList<Double>> readProcessedCSV(String directory){
        return Arrays.stream(readCSV(directory + "processedData.csv", true))
                .map(x -> Arrays.stream(x)
                        .map(Double::parseDouble)
                        .collect(Collectors.toCollection(ArrayList::new)))
                .collect(Collectors.toCollection(ArrayList::new));
    }

    private ArrayList<ArrayList<String>> readNormalizationFactors(String directory){
        return Arrays.stream(readCSV(directory, false))
                .map(x -> Arrays.stream(x)
                        .collect(Collectors.toCollection(ArrayList::new)))
                .collect(Collectors.toCollection(ArrayList::new));
    }

    void writeOutputCSV(String directory, String[][] data){
        String[][] helper = transpose2dArray(data);
        ArrayList<String[]> toWrite = new ArrayList<>(Arrays.asList(helper));
        writeCSV(directory, toWrite);
    }

    ArrayList<ArrayList<Double>> getTestingData(String testingDirectory, String processedDirectory, String dependentVariable){
        String[][] rawData = readCSV(testingDirectory, true);
        ArrayList<ArrayList<String>> normalizationFactors = readNormalizationFactors(processedDirectory + "normalizationFactors.csv");
        ArrayList<Double> ids = Arrays.stream(rawData[0])
                .skip(1)
                .mapToDouble(Double::parseDouble)
                .boxed()
                .collect(Collectors.toCollection(ArrayList::new));
        ArrayList<ArrayList<Double>> testingData = processData(false, normalizationFactors, rawData, dependentVariable);
        testingData.add(0, ids);
        return testingData;
    }

    private void calculateNormalizationFactors(ArrayList<ArrayList<String>> normalizationFactors, String[][] raw, String dependentVariable, boolean assumeStringsCategorical, ArrayList<String> otherCategoricalData, ArrayList<String> excludedData){
        for(String[] feild : raw){
            if(!excludedData.contains(feild[0]) && !feild[0].equals(dependentVariable)) {//ignore excluded data fields

                //handle + detect categorical variables
                if(otherCategoricalData.contains(feild[0]) ||
                        (assumeStringsCategorical &&
                             Arrays.stream(feild)
                                  .skip(1)
                                  .anyMatch(x -> !(x.matches("(-?\\d+(\\.\\d+)?)") || x.equals("NA")))
                        )
                ){
                    if(debugMode) System.out.println(" Feild " + feild[0] + " was found to be categorical");
                    ArrayList<String> curr = new ArrayList<>();
                    curr.add("Categorical");
                    curr.add(feild[0]);
                    for (int j = 1; j < feild.length; j++) {
                        if(!curr.contains(feild[j]))
                            curr.add(feild[j]);
                    }
                    normalizationFactors.add(curr);
                }

                else{
                    if(debugMode) System.out.println(" Feild " + feild[0] + " was found to be continuous");
                    ArrayList<String> curr = new ArrayList<>();
                    curr.add("Continuous");
                    curr.add(feild[0]);
                    ArrayList<Double> currData = Arrays.stream(feild)
                            .skip(1)
                            .filter(x -> !x.equals("NA"))
                            .map(Double::parseDouble)
                            .collect(Collectors.toCollection(ArrayList::new));
                    DescriptiveStatistics d = new DescriptiveStatistics(currData
                            .stream()
                            .mapToDouble(Double::doubleValue)
                            .toArray());
                    double lowerPercentile = d.getPercentile(25),
                            upperPercentile = d.getPercentile(75),
                            minimum = Double.MAX_VALUE,
                            maximum = Double.MIN_VALUE;
                    double IQR = upperPercentile - lowerPercentile;
                    for (int j = 0; j < currData.size(); j++) {
                        boolean remove = false;
                        if(REMOVE_OUTLIERS &&
                            ((currData.get(j)<lowerPercentile - 1.5 * IQR) ||
                            (currData.get(j)>upperPercentile + 1.5 * IQR))){
                            Scanner s = new Scanner(System.in);
                            System.out.println("Element of field " + curr.get(1) + " with ID " + (j+1) + " was found to be an outlier.");
                            System.out.println("Lower Percentile: " + lowerPercentile + " Upper Percentile: " + upperPercentile);
                            System.out.println("Value: " + currData.get(j));
                            System.out.println("Remove outlier? y/N");
                            remove = s.nextLine().equals("y");
                            if(remove) {
                                currData.remove(currData.get(j));
                                j++; //accounts for reindexing
                            }
                        }
                        if(!remove){
                            if(currData.get(j)<minimum) minimum = currData.get(j);
                            if(currData.get(j)>maximum) maximum = currData.get(j);
                        }
                    }
                    curr.add(minimum+"");
                    curr.add(maximum+"");
                    normalizationFactors.add(curr);
                }
            }
            else if(debugMode) System.out.println(" Feild " + feild[0] + " was skipped");
        }
        if(debugMode) printDoubleList(normalizationFactors);
    }

    private ArrayList<ArrayList<Double>> processData(boolean findDependencies, ArrayList<ArrayList<String>> normalizationFactors, String[][] rawData, String dependentVariable){
        ArrayList<String> currentFeilds = normalizationFactors.stream()
                .map(x -> x.get(1))
                .collect(Collectors.toCollection(ArrayList::new));
        ArrayList<ArrayList<Double>> processedData = new ArrayList<>();
        ArrayList<Double> dependentVariableData = new ArrayList<>(); //separated as it is not normalized or checked for dependency

        ArrayList<ArrayList<String>> processedFeilds = new ArrayList<>(); //just for testing
        processedFeilds.add(new ArrayList<>());
        processedFeilds.add(new ArrayList<>());

        for(String[] feild : rawData) {
            if (currentFeilds.contains(feild[0])) {//ignore excluded data fields
                if(debugMode) System.out.println("Processing Field " + feild[0]);

                for(ArrayList<String> nF : normalizationFactors){
                    if(feild[0].equals(nF.get(1))){
                        if(nF.get(0).equals("Categorical")){
                            if(debugMode) System.out.println(feild[0] + " was found to be categorical");
                            ArrayList<ArrayList<Double>> dummies = new ArrayList<>();
                            int numBits = (int)(Math.ceil(Math.log(nF.size()-2) / Math.log(2)));
                            if(debugMode) System.out.println("Numbits: " + numBits + " for " + (nF.size()-2) + " categories");
                            for(int i = 0; i<numBits; i++) {
                                dummies.add(new ArrayList<>());
                                processedFeilds.get(0).add(feild[0] + "_dummy"+(i+1));
                                processedFeilds.get(1).add(processedFeilds.get(0).size()+"");
                            }
                            for(int i = 1; i<feild.length; i++){
                                boolean found = false;
                                for(int j = 2; j<nF.size(); j++){
                                    if(feild[i].equals(nF.get(j))){
                                        found = true;
                                        char[] binary = Integer.toBinaryString(j-1).toCharArray();
                                        for(int k = 0; k<numBits; k++){
                                            if(k<binary.length)
                                                dummies.get(k).add(Double.parseDouble(binary[binary.length-1-k]+""));
                                            else
                                                dummies.get(k).add(0.0);
                                        }
                                    }
                                }
                                if(!found) {
                                    for (int k = 0; k < numBits; k++) {
                                        dummies.get(k).add(0.0);
                                    }
                                }
                            }
                            processedData.addAll(dummies);
                            if(debugMode) {
                                System.out.println("Dummies: ");
                                printDoubleDoubleList(dummies);
                                System.out.println("From: ");
                                printArray(feild);
                                System.out.println("Based on: ");
                                printList(nF);
                            }
                        }
                        else{
                            if(debugMode) System.out.println(feild[0] + " was found to be continuous");
                            ArrayList<Double> normField = new ArrayList<>();
                            double[] availableData = Arrays.stream(feild)
                                    .skip(1)
                                    .filter(x -> !x.equals("NA"))
                                    .mapToDouble(Double::parseDouble)
                                    .toArray();
                            double mean = Arrays.stream(availableData).sum() / availableData.length;
                            for(int i = 1; i<feild.length; i++){
                                if(feild[i].equals("NA")){
                                    feild[i] = mean+"";
                                }
                                normField.add((Double.parseDouble(feild[i])-Double.parseDouble(nF.get(2)))/
                                        (Double.parseDouble(nF.get(3))-Double.parseDouble(nF.get(2))));
                            }
                            processedData.add(normField);
                            processedFeilds.get(0).add(feild[0]);
                            processedFeilds.get(1).add(processedFeilds.get(0).size()+"");
                            if(debugMode) {
                                System.out.println("Processed Data: ");
                                printDList(normField);
                                System.out.println(" from ");
                                printArray(feild);
                            }
                        }
                    }
                }
            }
            //converts dependent variable to list
            else if(feild[0].equals(dependentVariable)){
                if(debugMode) System.out.println("Processing dependent variable " + feild[0]);
                    dependentVariableData = Arrays.stream(feild)
                            .skip(1)
                            .map(Double::parseDouble)
                            .collect(Collectors.toCollection(ArrayList::new));
                }
            else{
                if(debugMode) System.out.println("Skipping Field " + feild[0]);
            }
        }

        if(findDependencies){
            ArrayList<Integer> toRemove = new ArrayList<>();
            double[][] processedArray = new double[processedData.size()][];
            for(int i = 0; i<processedArray.length; i++){
                processedArray[i] = processedData.get(i)
                        .stream()
                        .mapToDouble(Double::valueOf)
                        .toArray();
            }
            System.out.println("Feilds: ");
            printDoubleList(processedFeilds);
            PearsonsCorrelation pearsonsCorrelation = new PearsonsCorrelation();

            for(int i = 0; i<processedData.size(); i++){
                ArrayList<String> correlations = new ArrayList<>();
                for(int j = i+1; j<processedData.size(); j++){
                    double correlationCoefficient = Math.abs(pearsonsCorrelation.correlation(processedArray[i], processedArray[j]));
                    if(correlationCoefficient>CORR_COEFF_CUTOFF){
                        correlations.add(processedFeilds.get(0).get(j));
                    }
                }
                if(!correlations.isEmpty() && !processedFeilds.get(0).get(i).contains("dummy")) {
                    System.out.println("The field " + processedFeilds.get(0).get(i)
                            + " was found to be highly correlated with the following fields:");
                    printList(correlations);
                }
            }
        }

        if(!dependentVariableData.isEmpty()) processedData.add(dependentVariableData);
        return processedData;
    }

    private void writeCSV(String directory, ArrayList<String[]> toWrite){
        CSVWriter csvWriter = null;
        try {
            if(debugMode) System.out.println("Writing file at " + directory);
            csvWriter = new CSVWriter(new FileWriter(directory));
            csvWriter.writeAll(toWrite);
        }catch (java.io.IOException e){
            System.out.println("An error occured when attempting to write a file.");
            System.out.println("Message: \n" + e.getMessage());
            System.out.println("StackTrace:");
            e.printStackTrace();
        }finally {
            try {
                if(csvWriter!=null)
                    csvWriter.close();
            } catch (java.io.IOException e) {
                System.out.println("An error occured when attempting to close a file connection.");
                System.out.println("Message: \n" + e.getMessage());
                System.out.println("StackTrace:");
            }
        }
    }

    private String[][] readCSV(String directory, boolean transpose){
        ArrayList<String[]> temp = new ArrayList<>();
        try {
            CSVReader reader = new CSVReader(new FileReader(directory));
            System.out.println("Reading raw data from " + directory + "... ");
            int column = 0;
            String[] curr;
            while ((curr = reader.readNext()) != null) {
                temp.add(curr);
                column++;
            }
        }
        catch (FileNotFoundException e){
            System.err.println("Error: CSV File not found");
            e.printStackTrace();
            System.exit(1);
        }
        catch (IOException e){
            System.err.println("Error: OpenCSV IOException Caught");
            e.printStackTrace();
            System.exit(1);
        }

        String[][] raw = new String[temp.size()][];
        for(int i = 0; i < temp.size(); i++){
            raw[i] = temp.get(i);
        }

        if(transpose) raw = transpose2dArray(raw);
        return raw;
    }

    private void printMatrix(RealMatrix m){
        for(int i = 0; i < m.getColumnDimension(); i++){
            System.out.print("Column " + i + ": ");
            double[] column = m.getColumn(i);
            for(double d : column){
                System.out.print(" " + d + " ");
            }
            System.out.println();
        }
    }

    private void printDoubleDoubleList(ArrayList<ArrayList<Double>> list){
        for(ArrayList<Double> arr : list){
            System.out.print("Column: ");
            printDList(arr);
        }
    }

    private void printDList(ArrayList<Double> list){
        for(Double d : list)
            System.out.print(d + " ");
        System.out.print("\n");
    }
    private void printDoubleList(ArrayList<ArrayList<String>> list){
        for(int i = 0; i < list.size(); i++){
            System.out.print("Column " + i + ": ");
            printList(list.get(i));
        }
    }

    private void printList(ArrayList<String> list){
        for(String s : list){
            System.out.printf("%-21s", " " + s + " ");
        }
        System.out.print("\n");
    }

    private void printReadFile(String[][] raw){
        for(int i = 0; i < raw.length; i++){
            System.out.print("Column " + i + ": ");
            printArray(raw[i]);
        }
    }

    private void printArray(String[] raw){
        for(int i = 0; i < raw.length; i++){
            System.out.print(" " + raw[i] + " ");
        }
        System.out.println();
    }

    private static String[][] transpose2dArray(String[][] array){
        int m = array.length;
        int n = array[0].length;

        String[][] transposedArray = new String[n][m];

        for(int x = 0; x < n; x++)
            for(int y = 0; y < m; y++)
                transposedArray[x][y] = array[y][x];

        return transposedArray;
    }
}
