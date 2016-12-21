package freagle.searching.feature;

import freagle.searching.bean.Document;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import scala.Tuple2;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
 * Created by freagle on 16-12-14.
 */
public class Example {
    public static void main(String[] args) throws Exception {
        Extractor extractor = new Extractor();
        JavaRDD<Document> result = extractor.extract();

        JavaPairRDD<String, Set<String>> invertedDocs = result.flatMapToPair(doc -> {
            List<Tuple2<String, Set<String>>> tupleList = new LinkedList<>();
            for (String word : doc.getWords()) {
                Set<String> docID = new HashSet<>();
                docID.add(doc.getId());
                tupleList.add(new Tuple2<>(word, docID));
            }
            return tupleList.iterator();
        }).reduceByKey((docID1, docID2) -> {
            for (String docID : docID2) {
                docID1.add(docID);
            }
            return docID1;
        });

        List<Tuple2<String, Set<String>>> takes = invertedDocs.take(10);

        for (Tuple2<String, Set<String>> take : takes) {
            System.out.println(take._1() + ":" + take._2());
        }


        extractor.getSpark().stop();

    }
}
