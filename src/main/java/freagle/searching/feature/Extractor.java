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

import java.io.Serializable;
import java.util.List;

/**
 * Created by freagle on 16-12-9.
 */
public class Extractor implements Serializable {
    private String docsPath;
    private SparkSession spark;
    private HashingTF hashingTF = new HashingTF();
    private IDFModel idfModel;

    public Extractor(){
        this.docsPath = "target/classes/20_newsgroups/alt.atheism";
        this.spark = SparkSession
                .builder()
                .appName("Searching")
                .master("local[*]")
                .getOrCreate();
    }

    public Extractor(String docsPath){
        this.docsPath = docsPath;
        this.spark = SparkSession
                .builder()
                .appName("Searching")
                .master("local[*]")
                .getOrCreate();
    }

    public Extractor(String docsPath, SparkSession spark){
        this.docsPath = docsPath;
        this.spark = spark;
    }

    public HashingTF getHashingTF() {
        return hashingTF;
    }

    public void setHashingTF(HashingTF hashingTF) {
        this.hashingTF = hashingTF;
    }

    public String getDocsPath() {
        return docsPath;
    }

    public void setDocsPath(String docsPath) {
        this.docsPath = docsPath;
    }

    public SparkSession getSpark() {
        return spark;
    }

    public void setSpark(SparkSession spark) {
        this.spark = spark;
    }

    /**
     * 从给定的文档集计算出每篇文档的tf-idf
     * @return 包含计算结果的Java bean的RDD
     */
    public JavaRDD<Document> extract(){
        JavaRDD<Document> tokenizedDocs = tokenizeDocs(createDataset());
        calIDFModel(tokenizedDocs);
        return calTfIdf(tokenizedDocs);
    }

    private Dataset<Row> createDataset(){
//        读取文档生成RDD
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        JavaPairRDD<String, String> docMaps = jsc.wholeTextFiles(this.docsPath);
        JavaRDD<Document> docs = docMaps.map(docMap -> {
            Document doc = new Document();
            String[] splits = docMap._1.split("/");
            doc.setId(splits[splits.length - 1]);
            doc.setContent(docMap._2());
            return doc;
        });
//        将RDD转化为dataFrame
        return spark.createDataFrame(docs, Document.class);
    }

    private JavaRDD<Document> tokenizeDocs(Dataset<Row> docsDF){
//        完成分词和去除停用词
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

        return tokenizedDocs;
    }

    private void calIDFModel(JavaRDD<Document> tokenizedDocs){
//        计算tf-idf
        HashingTF hashingTF = new HashingTF();

//        先计算idf model
        JavaRDD<List<String>> docsWords = tokenizedDocs.map(doc -> doc.getWords());
        JavaRDD<Vector> tf = hashingTF.transform(docsWords);
        idfModel = new IDF().fit(tf);
    }

    private JavaRDD<Document> calTfIdf(JavaRDD<Document> tokenizedDocs){
//        如果还未计算idf模型，先计算idf模型
        if (this.idfModel == null) calIDFModel(tokenizedDocs);

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

        return result;
    }

}
