/*
Change log
----------
2015-08-13 : Initial release
*/

import java.awt.*;
import java.awt.geom.*;
import java.awt.event.*;
import java.awt.image.*;
import java.io.*;
import java.util.*;
import java.security.*;
import javax.swing.*;
import javax.imageio.*;

public class PopulationMappingVis {
    // tuning constants
    final static int MIN_SZ = 50, MAX_SZ = 500;
    final static double BASE_PROB = 0.5;
    final static double CELL_PROB = 0.25;
    final static double SURROUND_PROB = 0.2;
    // ----------------------------------------------------------------------------------
    Random r;
    int width, height;                 // size of the map
    int total;
    boolean[][] land;
    int landArea = 0;
    String[] map;
    int[][] pop;
    int target;
    int queryCount = 0;
    int[] centers = new int[0];
    int[] centersScale = new int[0];
    boolean failedQueries = false;

    class Query
    {
        int x1,y1,x2,y2;
        public Query(int x1, int y1, int x2, int y2) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
        }
    }
    ArrayList<Query> queries = new ArrayList<Query>();

    int maxpopoverall = 0;
    String[] returnMap;
    // ----------------------------------------------------------------------------------
    private int centerDistance(int y, int x) {
        int best = width + height;
        for (int i = 0; i < centers.length; i++) {
            int yy = centers[i] / 10000;
            int xx = centers[i] % 10000;
            best = Math.min(best, (Math.abs(y - yy) + Math.abs(x - xx)) / centersScale[i]);
        }
        return Math.max(best, 1);
    }
    // ----------------------------------------------------------------------------------
private void generateMap() {
  int w = 6;
  int h = 6;
  land = new boolean[h][w];
  for (int i = 1; i < 5; i++) for (int j = 1; j < 5; j++) land[i][j] = r.nextDouble() < BASE_PROB;
  while (w < width || h < height) {
    boolean[][] newland = new boolean[h * 2][w * 2];
    for (int y = 0; y < h * 2; y++) for (int x = 0; x < w * 2; x++) {
      double prob = BASE_PROB + (land[y / 2][x /2] ? CELL_PROB : -CELL_PROB);
      if (y % 2 == 0 && y > 0) prob += (land[y / 2 - 1][x / 2] ? SURROUND_PROB : -SURROUND_PROB);
      if (y % 2 == 1 && y < h - 1) prob += (land[y / 2 + 1][x / 2] ? SURROUND_PROB : -SURROUND_PROB);
      if (x % 2 == 0 && x > 0) prob += (land[y / 2][x / 2 - 1] ? SURROUND_PROB : -SURROUND_PROB);
      if (x % 2 == 1 && x < w - 1) prob += (land[y / 2][x / 2 + 1] ? SURROUND_PROB : -SURROUND_PROB);
      newland[y][x] = r.nextDouble() < prob;
    }
    land = newland;
    w *= 2;
    h *= 2;
  }
  map = new String[height];
  for (int y = 0; y < height; y++) {
    map[y] = "";
    for (int x = 0; x < width; x++)
      if (land[y][x]) {
        map[y] += "X";
        landArea++;
      } else {
        map[y] += ".";
      }
  }
}
    // ----------------------------------------------------------------------------------
private void generateCenters() {
  int numCenters = 5 + r.nextInt((width + height) / 100);
  centers = new int[numCenters];
  centersScale = new int[numCenters];
  for (int i = 0; i < numCenters; i++) {
    while (centers[i] == 0) {
      int y = r.nextInt(height);
      int x = r.nextInt(width);
      if (land[y][x]) {
        centers[i] = y * 10000 + x;
        centersScale[i] = 1 + r.nextInt(5);
      }
    }
  }
}
    // ----------------------------------------------------------------------------------
public void generateTestCase(long seed) {
  r = new Random(seed);
  width = r.nextInt(MAX_SZ - MIN_SZ + 1) + MIN_SZ;
  height = r.nextInt(MAX_SZ - MIN_SZ + 1) + MIN_SZ;
  if (seed < 3) {
    width = 50 + (int)seed * 10;
    height = 50 + (int)seed * 10;
  }
  pop = new int[height][width];
  generateMap();
  generateCenters();
  total = 0;
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
      if (land[y][x]) {
        int maxPop = (width + height) * 8 / centerDistance(y, x);
        total += pop[y][x] = r.nextInt(maxPop + 1);
        maxpopoverall = Math.max(maxpopoverall, pop[y][x]);
      }
  target = 97 - (int)Math.sqrt(r.nextInt(96 * 96));

  if (debug) {
    System.out.println("Width = " + width + "\nHeight = " + height + "\nTotal Population = " + total + "\nTarget = " + target + "%\nLand Area = " + landArea + "\n");
    for (int y = 0; y < height; y++) {
      System.out.println(map[y]);
    }
  }
}
    // ----------------------------------------------------------------------------------
public int queryRegion(int x1, int y1, int x2, int y2) {
  if (x1 > x2 || y1 > y2 || x1 < 0 || x2 > width - 1 || y1 < 0 || y2 > height - 1) {
    failedQueries = true;
    return -1;
  }
  queryCount++;
  queries.add(new Query(x1,y1,x2,y2));
  int ret = 0;
  for (int y = y1; y <= y2; y++) for (int x = x1; x <= x2; x++)
    ret += pop[y][x];
  return ret;
}
    // ----------------------------------------------------------------------------------
public double runTest(String test) {
  long seed = Long.parseLong(test);
  generateTestCase(seed);
  if (vis) {
    returnMap = new String[0];
    jf.setSize(width*scale+50,height*scale+40);
    jf.setVisible(true);
    draw();
  }
  String[] ret = mapPopulation(target, map, total);
  if (failedQueries) {
    addFatalError("One or more issued queries contained invalid parameters.");
    return -1.0;
  }
  if (ret.length != height) {
    addFatalError("Return array length must be the same as the initial map height.");
    return -1.0;
  }
  for (int i = 0; i < height; i++) {
    if (ret[i].length() != width) {
      addFatalError("Element " + i + " of return array length must be the same as the initial map width.");
      return -1.0;
    }
  }
  if (vis) {
    returnMap = ret;
    try { Thread.sleep(50);}
    catch (Exception e) { e.printStackTrace(); }
    draw();
  }
  long maxTarget = (long)target * (long)total / 100;
  long counted = 0;
  int area = 0;
  boolean[][] count = new boolean[height][width];
  for (int y = 0; y < ret.length; y++)
    for (int x = 0; x < ret[y].length(); x++) {
      if (ret[y].charAt(x) == 'X' && land[y][x]) {
        counted += pop[y][x];
        area++;
      }
    }
  if (counted > maxTarget) {
    addFatalError("Selected areas count too much population.");
    return -1;
  }
  addFatalError("Area = "+area+"\nQueries = "+queryCount);
  return 1.0 * area * Math.pow(0.996, queryCount);
}
// ------------- server part ------------------------------------------------------------
    public String checkData(String test) {
        return "";
    }
    // ----------------------------------------------------------------------------------
    public String displayTestCase(String test) {
        long seed = Long.parseLong(test);
        generateTestCase(seed);
        String st = "Width = " + width + "\nHeight = " + height + "\nTotal Population = " + total + "\nTarget = " + target + "%\nLand Area = " + landArea + "\n";
        if (height < 100 && width < 100) {
            for (int y = 0; y < height; y++) {
                st += map[y] + "\n";
            }
        }
        return st;
    }
    // ----------------------------------------------------------------------------------
    public double[] score(double[][] raw) {
        double[] ret = new double[raw.length];
        double[] best = new double[raw[0].length];
        for (int i = 0; i < raw.length; i++)
            for (int j = 0; j < raw[i].length; j++)
                best[j] = Math.max(best[j], raw[i][j]);
        for (int i = 0; i < raw.length; i++)
            for (int j = 0; j < raw[i].length; j++)
                if (raw[i][j] > 0 && best[j] > 0)
                    ret[i] += raw[i][j] / best[j];
        for (int i = 0; i < raw.length; i++) {
            ret[i] /= best.length;
            ret[i] *= 1000000.0;
        }
        return ret;
    }
// ------------- visualization part -----------------------------------------------------
    static int scale;
    static String exec;
    static boolean debug, vis;
    static Process proc;
    static int del;
    InputStream is;
    OutputStream os;
    BufferedReader br;
    JFrame jf;
    Vis v;
    // ----------------------------------------------------------------------------------
    String[] mapPopulation(int maxPercentage, String[] map, int totalPopulation) {
      int i;
      String[] ret = new String[0];
      try {
        if (exec != null)
        {   // imitate passing params
            StringBuffer sb = new StringBuffer();
            sb.append(maxPercentage).append('\n');
            sb.append(height).append('\n');
            for (i = 0; i < height; ++i)
                sb.append(map[i]).append('\n');
            sb.append(totalPopulation).append('\n');
            os.write(sb.toString().getBytes());
            os.flush();

            // imitate queries - they start with "?" in separate line followed by a line with 4 params
            String s;
            int row;
            while ((s = br.readLine()).equals("?")) {
                // get params of next call
                String[] params = br.readLine().split(" ");
                if (params.length != 4) {
                    // invalid # of arg
                    return ret;
                }
                int[] paramsInt = new int[4];
                try {
                    for (i = 0; i < 4; ++i) {
                        paramsInt[i] = Integer.parseInt(params[i]);
                    }
                } catch (NumberFormatException e) {
                    // failed to convert query arg to ints
                    return ret;
                }
                long queryRes = queryRegion(paramsInt[0], paramsInt[1], paramsInt[2], paramsInt[3]);
                if (queryRes < 0) {
                    // other kind of invalid arg
                    return ret;
                }
                // and return the result to the solution
                os.write((queryRes + "\n").getBytes());
                os.flush();
            }

            int Nret = Integer.parseInt(s);
            ret = new String[Nret];
            for (i = 0; i < Nret; ++i)
                ret[i] = br.readLine();
        }
        return ret;
      } catch (IOException e) {
        return ret;
      }
    }
    // ----------------------------------------------------------------------------------
    void draw() {
        if (!vis) return;
        v.repaint();
        try { Thread.sleep(del); }
        catch (Exception e) { };
    }
    // ----------------------------------------------------------------------------------
    public class Vis extends JPanel implements WindowListener {
        public void paint(Graphics gr) {
            int i, j;
            BufferedImage bi = new BufferedImage(width * scale + 150, height * scale + 1, BufferedImage.TYPE_INT_RGB);
            Graphics2D g2 = (Graphics2D)bi.getGraphics();
            // background
            g2.setColor(new Color(0xEEEEEE));
            g2.fillRect(0, 0, width * scale + 150, height * scale + 1);
            g2.setColor(new Color(0xAAAAAA));
            g2.fillRect(0, 0, width * scale, height * scale);
            boolean highlight = (returnMap.length > 0);
            // map cells: make oceans light-blue, population in shades of red, selected areas of population in green
            // don't highlight selected oceans, as they don't affect # anyways
            // lighter shade = less population density
            for (i = 0; i < height; ++i)
            for (j = 0; j < width; ++j) {
                if (!land[i][j]) {
                    g2.setColor(new Color(0x00CCCC));
                } else {
                    int mult = ((maxpopoverall - pop[i][j]) * 255) / maxpopoverall;
                    int c;
                    if (!highlight || returnMap[i].charAt(j) != 'X') {
                        c = 0xFF0000 + 0x000101 * mult;
                    } else {
                        c = 0x770077 + 0x0100 * mult;
                    }
                    g2.setColor(new Color(c));
                }
                g2.fillRect(j * scale + 1, i * scale + 1, scale, scale);
            }

            g2.setColor(new Color(0x00AAFF));
            for (Query q : queries) {
                g2.drawRect(q.x1 * scale, q.y1 * scale, (q.x2-q.x1+1)* scale+1, (q.y2-q.y1+1)* scale+1);
            }

            gr.drawImage(bi, 0, 0, width * scale + 150, height * scale + 1, null);
//            try { ImageIO.write(bi, "png", new File("1.png")); } catch (Exception e) {};
        }
        public Vis() {
            jf.addWindowListener(this);
        }
        //WindowListener
        public void windowClosing(WindowEvent e){
            if(proc != null)
                try { proc.destroy(); }
                catch (Exception ex) { ex.printStackTrace(); }
            System.exit(0);
        }
        public void windowActivated(WindowEvent e) { }
        public void windowDeactivated(WindowEvent e) { }
        public void windowOpened(WindowEvent e) { }
        public void windowClosed(WindowEvent e) { }
        public void windowIconified(WindowEvent e) { }
        public void windowDeiconified(WindowEvent e) { }
    }
    // ----------------------------------------------------------------------------------
    public PopulationMappingVis(String seed) {
      try {
        // interface for runTest
        if (vis)
        {   jf = new JFrame();
            v = new Vis();
            jf.getContentPane().add(v);
        }
        if (exec != null) {
            try {
                Runtime rt = Runtime.getRuntime();
                proc = rt.exec(exec);
                os = proc.getOutputStream();
                is = proc.getInputStream();
                br = new BufferedReader(new InputStreamReader(is));
                new ErrorReader(proc.getErrorStream()).start();
            } catch (Exception e) { e.printStackTrace(); }
        }
        System.out.println("Score = "+runTest(seed));
        if (proc != null)
            try { proc.destroy(); }
            catch (Exception e) { e.printStackTrace(); }
      }
      catch (Exception e) { e.printStackTrace(); }
    }
    // ----------------------------------------------------------------------------------
    public static void main(String[] args) {
        String seed = "1";
        vis = true;
        debug = false;
        del = 1000;
        scale = 2;
        for (int i = 0; i<args.length; i++)
        {   if (args[i].equals("-seed"))
                seed = args[++i];
            if (args[i].equals("-exec"))
                exec = args[++i];
            if (args[i].equals("-novis"))
                vis = false;
            if (args[i].equals("-debug"))
                debug = true;
            if (args[i].equals("-delay"))
                del = Integer.parseInt(args[++i]);
            if (args[i].equals("-scale"))
                scale = Integer.parseInt(args[++i]);
        }
        PopulationMappingVis bw = new PopulationMappingVis(seed);

    }
    // ----------------------------------------------------------------------------------
    void addFatalError(String message) {
        System.out.println(message);
    }
}

class ErrorReader extends Thread{
    InputStream error;
    public ErrorReader(InputStream is) {
        error = is;
    }
    public void run() {
        try {
            byte[] ch = new byte[50000];
            int read;
            while ((read = error.read(ch)) > 0)
            {   String s = new String(ch,0,read);
                System.out.print(s);
                System.out.flush();
            }
        } catch(Exception e) { }
    }
}

