package freagle.searching.bean;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by freagle on 16-12-20.
 */
public class Document implements Serializable {
    private String id;
    private String content;
    private List<String> words;
    private Map<String, Integer> wordsHash;
    private Map<Integer, Double> wordsTfIdf;

    public List<String> getWords() {
        if (this.words == null){
            this.setWords(new ArrayList<String>());
        }
        return words;
    }

    public void setWords(List<String> words) {
        this.words = words;
    }

    public String getId() {

        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public Map<String, Integer> getWordsHash() {
        if (this.wordsHash == null) {
            this.setWordsHash(new HashMap<String, Integer>(this.getWords().size()));
        }
        return wordsHash;
    }

    public void setWordsHash(Map<String, Integer> wordsHash) {
        this.wordsHash = wordsHash;
    }

    public Map<Integer, Double> getWordsTfIdf() {
        if (this.wordsTfIdf == null){
            this.setWordsTfIdf(new HashMap<Integer, Double>(this.getWords().size()));
        }
        return wordsTfIdf;
    }

    public void setWordsTfIdf(Map<Integer, Double> wordsTfIdf) {
        this.wordsTfIdf = wordsTfIdf;
    }

    public double getTfIdfByWord(String word){
        return this.getWordsTfIdf().get(this.getWordsHash().get(word));
    }
}
