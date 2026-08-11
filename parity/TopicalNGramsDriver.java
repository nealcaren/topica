// Java driver: run Java MALLET's TopicalNGrams (Wang, McCallum & Wei 2007) and dump
// the full per-token Gibbs state, for statistical parity with topica.TopicalNGrams.
//
// Input file, one document per line: whitespace-tokenized tokens (already cleaned;
// every adjacent in-vocab pair is bigram-eligible, matching topica's default rule).
// Output <out_state>: MALLET's printState —
//   header "#doc pos typeindex type bigrampossible? topic bigram"
//   then one line per token: "<doc> <pos> <typeindex> <type> <bigrampossible> <topic> <gram>"
// from which the parity harness reconstructs the unigram topic-word matrix, the
// doc-topic matrix, and the per-topic phrases.
//
// Usage: TopicalNGramsDriver <input> <numTopics> <iterations> <seed> \
//          <alphaSum> <beta> <gamma> <delta> <delta1> <delta2> <out_state>

import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.TokenSequence2FeatureSequenceWithBigrams;
import cc.mallet.topics.TopicalNGrams;
import cc.mallet.types.Alphabet;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import cc.mallet.util.Randoms;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

public class TopicalNGramsDriver {
    public static void main(String[] args) throws Exception {
        String input = args[0];
        int numTopics = Integer.parseInt(args[1]);
        int iterations = Integer.parseInt(args[2]);
        int seed = Integer.parseInt(args[3]);
        double alphaSum = Double.parseDouble(args[4]);
        double beta = Double.parseDouble(args[5]);
        double gamma = Double.parseDouble(args[6]);
        double delta = Double.parseDouble(args[7]);
        double delta1 = Double.parseDouble(args[8]);
        double delta2 = Double.parseDouble(args[9]);
        String outState = args[10];

        // The bigram alphabet must be supplied to the pipe; the unigram alphabet is
        // created by the pipe. This pipe makes every adjacent token pair a candidate
        // bigram (biIndex != -1 except at position 0).
        Alphabet biAlphabet = new Alphabet();
        Pipe pipe = new TokenSequence2FeatureSequenceWithBigrams(biAlphabet);
        InstanceList instances = new InstanceList(pipe);

        BufferedReader br = new BufferedReader(new FileReader(input));
        String line;
        while ((line = br.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            TokenSequence ts = new TokenSequence();
            for (String t : line.trim().split("\\s+")) ts.add(new Token(t));
            instances.addThruPipe(new Instance(ts, null, null, null));
        }
        br.close();

        TopicalNGrams model =
            new TopicalNGrams(numTopics, alphaSum, beta, gamma, delta, delta1, delta2);
        model.estimate(instances, iterations, 0, 0, null, new Randoms(seed));
        model.printState(new File(outState));
    }
}
