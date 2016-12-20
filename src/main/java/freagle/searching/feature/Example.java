package freagle.searching.feature;

import freagle.searching.bean.Document;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.List;

/**
 * Created by freagle on 16-12-14.
 */
public class Example {
    public static void main(String[] args) throws Exception {
//        File files = FileSystems.getDefault().getPath("target/classes/20_newsgroups/alt.atheism").toFile();
//        for (File file : files.listFiles()) {
//            System.out.println(file.toPath().toString());
//        }

//        SparkSession spark = SparkSession
//                .builder()
//                .appName("Java Spark SQL basic example")
//                .master("local[*]")
//                .getOrCreate();
//
//        List<Row> data = Arrays.asList(
//                RowFactory.create(0.0, "Hi I heard about Spark"),
//                RowFactory.create(0.0, "I wish Java could use case classes"),
//                RowFactory.create(1.0, "Logistic regression models are neat")
//        );
//        StructType schema = new StructType(new StructField[]{
//                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
//                new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
//        });
//        Dataset<Row> sentenceData = spark.createDataFrame(data, schema);
//        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
//        Dataset<Row> wordsData = tokenizer.transform(sentenceData);
//        HashingTF hashingTF = new HashingTF()
//                .setInputCol("words")
//                .setOutputCol("rawFeatures");
//        Dataset<Row> featurizedData = hashingTF.transform(wordsData);
//        System.out.println(hashingTF.explainParams());
//// alternatively, CountVectorizer can also be used to get term frequency vectors
//
//        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
//        IDFModel idfModel = idf.fit(featurizedData);
//        Dataset<Row> rescaledData = idfModel.transform(featurizedData);
//        for (Row r : rescaledData.select("words", "rawFeatures", "features", "label").takeAsList(3)) {
//            Vector features = r.getAs(2);
//            Double label = r.getDouble(3);
//            List<String> words = r.getList(0);
//            System.out.println(r.getAs(1).toString());
//            System.out.println(String.join(",", words));
//            System.out.println(features);
//            System.out.println(label);
//        }
        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .master("local[*]")
                .getOrCreate();

//        读取文档生成RDD
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        JavaPairRDD<String, String> docMaps = jsc.wholeTextFiles("target/classes/20_newsgroups/alt.atheism");
        JavaRDD<Document> docs = docMaps.map(docMap -> {
            Document doc = new Document();
            String[] splits = docMap._1.split("/");
            doc.setId(splits[splits.length - 1]);
            doc.setContent(docMap._2());
            return doc;
        });

//        将RDD转化为dataFrame，完成分词和去除停用词
        Dataset<Row> docsDF = spark.createDataFrame(docs, Document.class);

        RegexTokenizer regexTokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("tokenizedWords")
                .setPattern("\\W");
        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol("tokenizedWords")
                .setOutputCol("filteredWords");
        Dataset<Row> tokenizedDocsDF = remover.transform(regexTokenizer.transform(docsDF));

//        将分词过的dataFrame转换为RDD
        JavaRDD<Row> tokenizedDocRows = tokenizedDocsDF.select("id", "content", "filteredWords").toJavaRDD();

        JavaRDD<Document> tokenizedDocs = tokenizedDocRows.map(row -> {
            Document doc = new Document();
            doc.setId(row.getString(0));
            doc.setContent(row.getString(1));
            doc.setWords(row.getList(2));
            return doc;
        });

//        计算tf-idf
        HashingTF hashingTF = new HashingTF();

//        先计算idf model
        JavaRDD<List<String>> docsWords = tokenizedDocs.map(doc -> doc.getWords());
        JavaRDD<Vector> tf = hashingTF.transform(docsWords);
        IDFModel idfModel = new IDF().fit(tf);

//        再计算每个文档的tf-idf
        JavaRDD<Document> result = tokenizedDocs.map(doc -> {
            Vector tfIdfVec = idfModel.transform(hashingTF.transform(doc.getWords()));

            for (String word : doc.getWords()) {
                int i = hashingTF.indexOf(word);
                doc.getWordsHash().put(word, i);
                doc.getWordsTfIdf().put(i, tfIdfVec.apply(i));
            }
            return doc;
        });

        List<Document> aDoc = result.take(1);

        for (Document document : aDoc) {
            System.out.println(document.getId());
            System.out.println(document.getWords());
            System.out.println(document.getWordsHash());
            System.out.println(document.getWordsTfIdf());
        }

        spark.stop();

    }
}
