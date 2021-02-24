package ehu.weka;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.FileWriter;

public class IragarpenaEgin {

    public static void main(String[] args) throws Exception {

        // TODO Auto-generated method stub

        if (args.length != 3) {

            System.out.println("\nUsage: IragarpenaEgin </path/NB.model> </path/test_blind.arff> </path/test_predictions.txt>\n");

        } else {

            String modeloa = args[0];
            String datuak = args[1];
            String txtpredict = args[2];

            //Lehenengo modeloa lortuko dugu
            Classifier cls = (Classifier) weka.core.SerializationHelper.read(modeloa);
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(datuak);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);

            Evaluation eval = new Evaluation(data);
            FileWriter fw = new FileWriter(txtpredict);
            eval.evaluateModel(cls, data);

            fw.write("/////////////////////////PREDIKZIOAK/////////////////////////\n\n\n");
            for(Prediction p:  eval.predictions()) {
                fw.write("\n Iragarpena : "+data.attribute(data.numAttributes()-1).value((int) p.predicted()) + " eta balio erreala : " +data.attribute(data.numAttributes()-1).value((int) p.actual()));
            }
            fw.write("\n"+eval.toMatrixString());
            fw.write(String.valueOf(eval.weightedFMeasure()));
            fw.close();
        }
    }

}
