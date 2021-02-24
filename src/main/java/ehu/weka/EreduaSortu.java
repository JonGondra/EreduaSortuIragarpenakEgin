package ehu.weka;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

public class EreduaSortu {

    public static void main(String[] args) throws Exception {

        if (args.length != 3) {
            System.out.println("\nUsage: EreduaSortu </path/data.arff> </path/NB.model> </path/KalitatearenEstimazioa.txt>\n");
        }
        else{
            String datuak = args[0];
            String modeloemaitza = args[1];
            String txt = args[2];
            java.util.Date date = new java.util.Date();

            //datuak kargatu
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(datuak);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);

            //txt sortu
            FileWriter fw = new FileWriter(txt);
            fw.write("/////////////////////////KALITATEAREN ESTIMAZIOA/////////////////////////\n\n\n");
            fw.write("\t\t\t"+date+"\n\n\n");
            fw.write("/////////////////////////Jasotako argumentuak/////////////////////////\n\n");
            fw.write("Jasotako datuak : "+datuak);
            fw.write("\nGordetako modeloa : "+modeloemaitza);
            fw.write("\nFitxategi hau : "+txt+"\n");


            fw.write("\n:::::::::::::::::::::::::K-FOLD CROSS VALIDATION:::::::::::::::::::::::::\n");
            NaiveBayes nv = new NaiveBayes();
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(nv,data,10,new Random(1));
            fw.write("\n"+eval.toMatrixString()+"\n");

            fw.write("\n:::::::::::::::::::::::::HOLD OUT 70%:::::::::::::::::::::::::\n");
            Randomize randomize = new Randomize();
            randomize.setInputFormat(data);
            data = Filter.useFilter(data,randomize);

            //%70 hartu (train)
            RemovePercentage filterRemoveTrain = new RemovePercentage();
            filterRemoveTrain.setInputFormat(data);
            filterRemoveTrain.setPercentage(70);
            filterRemoveTrain.setInvertSelection(true);
            Instances train = Filter.useFilter(data,filterRemoveTrain);
            train.setClassIndex(train.numAttributes() - 1);

            //%30 hartu (test)
            RemovePercentage filterRemoveTest = new RemovePercentage();
            filterRemoveTest.setInputFormat(data);
            filterRemoveTest.setPercentage(70);
            Instances test = Filter.useFilter(data,filterRemoveTest);
            test.setClassIndex(test.numAttributes() - 1);

            eval = new Evaluation(train);
            nv = new NaiveBayes();
            nv.buildClassifier(train);
            weka.core.SerializationHelper.write(modeloemaitza, nv);
            eval.evaluateModel(nv,test);
            fw.write("\n"+eval.toMatrixString()+"\n");

            fw.close();
    }
}
}
