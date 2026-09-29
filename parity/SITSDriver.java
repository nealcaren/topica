import java.io.*;
import java.lang.reflect.Field;
import java.util.*;
import segmentation.parametric.sampler.AuthorShiftSampler;

/**
 * Driver for the parametric SITS Gibbs sampler of Rossiter's fork
 * (github.com/erossiter/sits, Apache-2.0), used by parity/sits_compare.py.
 *
 * The fork's two static RNGs (core.AbstractSampler.rand, util.SamplerUtils.rand)
 * are fixed-seed; this driver reseeds both by reflection so chains differ by seed,
 * which the fork itself cannot do (it varies chains only through I).
 *
 * args: wordsFile authorsFile V J K alpha beta gamma burnIn maxIter I seed outDir
 * The sampler writes every post-burn-in shift vector to
 * outDir/<samplerName>/all_sampled_shift_asgn.txt; the driver also writes
 * phi.txt / pi.txt and records the sampler folder and wall time in outDir/run.txt.
 */
public class SITSDriver {
    public static void main(String[] a) throws Exception {
        String wordsF = a[0], authorsF = a[1];
        int V = Integer.parseInt(a[2]), J = Integer.parseInt(a[3]), K = Integer.parseInt(a[4]);
        double alpha = Double.parseDouble(a[5]), beta = Double.parseDouble(a[6]), gamma = Double.parseDouble(a[7]);
        int burn = Integer.parseInt(a[8]), maxIter = Integer.parseInt(a[9]), I = Integer.parseInt(a[10]);
        long seed = Long.parseLong(a[11]);
        String out = a[12]; if (!out.endsWith("/")) out += "/";
        // reseed both static RNGs (AbstractSampler.rand, SamplerUtils.rand)
        Class<?> as = Class.forName("core.AbstractSampler");
        Field f = as.getDeclaredField("rand"); f.setAccessible(true); f.set(null, new Random(seed));
        Class<?> su = Class.forName("util.SamplerUtils");
        Field g = su.getDeclaredField("rand"); g.setAccessible(true); g.set(null, new Random(seed + 1));
        // read words: header lines, then turns, blank line between conversations
        BufferedReader br = new BufferedReader(new FileReader(wordsF));
        int C = Integer.parseInt(br.readLine().trim()); br.readLine();
        List<List<int[]>> conv = new ArrayList<>(); List<int[]> cur = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null) {
            if (line.trim().isEmpty()) { if (!cur.isEmpty()) conv.add(cur); cur = new ArrayList<>(); continue; }
            String[] p = line.split("\t", -1);
            int n = Integer.parseInt(p[0].trim());
            int[] w = new int[n];
            if (n > 0) { String[] ws = p[1].trim().split(" "); for (int i = 0; i < n; i++) w[i] = Integer.parseInt(ws[i]); }
            cur.add(w);
        }
        if (!cur.isEmpty()) conv.add(cur);
        br.close();
        // authors: -1 separates conversations
        br = new BufferedReader(new FileReader(authorsF));
        List<List<Integer>> auth = new ArrayList<>(); List<Integer> ca = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            int x = Integer.parseInt(line.trim());
            if (x == -1) { auth.add(ca); ca = new ArrayList<>(); } else ca.add(x);
        }
        if (!ca.isEmpty()) auth.add(ca);
        br.close();
        int[][][] words = new int[conv.size()][][]; int[][] authors = new int[conv.size()][];
        for (int c = 0; c < conv.size(); c++) {
            words[c] = conv.get(c).toArray(new int[0][]);
            authors[c] = new int[auth.get(c).size()];
            for (int t = 0; t < authors[c].length; t++) authors[c][t] = auth.get(c).get(t);
            if (authors[c].length != words[c].length) throw new RuntimeException("mismatch conv " + c);
        }
        AuthorShiftSampler s = new AuthorShiftSampler();
        s.configure(out, words, authors, K, J, V, alpha, beta, gamma, burn, maxIter, 1000000000, I);
        new File(out + s.getSamplerName()).mkdirs();
        long t0 = System.nanoTime();
        s.sample();
        double secs = (System.nanoTime() - t0) / 1e9;
        String d = out + s.getSamplerName() + "/";
        s.outputPi(d + "pi.txt"); s.outputPhi(d + "phi.txt"); s.outputTheta(d + "theta.txt");
        s.outputLogLikelihoods(d + "loglikelihood.txt");
        PrintWriter pw = new PrintWriter(out + "run.txt"); pw.println(d); pw.println(secs); pw.close();
        System.out.println(d + " " + secs);
    }
}
