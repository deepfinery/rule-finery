package demo;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.kie.api.io.ResourceType;
import org.kie.api.runtime.KieSession;
import org.kie.internal.utils.KieHelper;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class Runner {

  static final ObjectMapper M = new ObjectMapper()
      .disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);

  @SuppressWarnings("unchecked")
  public static void main(String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println("Usage: java -jar drools-runner.jar <path/to.drl> <input.json or ->");
      System.exit(2);
    }
    String drlPath = args[0];
    String inputPath = args[1];

    String drl = Files.readString(Path.of(drlPath));
    KieHelper helper = new KieHelper().addContent(drl, ResourceType.DRL);
    KieSession ksession = helper.build().newKieSession();

    // globals
    List<String> fired = new ArrayList<>();
    Map<String,Object> out = new LinkedHashMap<>();
    ksession.setGlobal("fired", fired);
    ksession.setGlobal("out", out);

    // read facts JSON
    InputStream in = "-".equals(inputPath) ? System.in : new FileInputStream(inputPath);
    Map<String,Object> facts = M.readValue(in, Map.class);

    // Insert each transaction as its own fact for per-tx rules
    Object txs = facts.get("recent_tx");
    if (txs instanceof List) {
      for (Object o : (List<?>) txs) {
        if (o instanceof Map) {
          Map<String,Object> tx = new HashMap<>((Map<String,Object>)o);
          tx.put("__type","Transaction");
          ksession.insert(tx);
        }
      }
    }

    // Optional: capture full trace metadata (rule name/time/etc.)
    // (Keeping it simple: we only append rule names to 'fired' list.)

    ksession.insert(facts);
    ksession.fireAllRules();
    ksession.dispose();

    // Build output
    Map<String,Object> decision = new LinkedHashMap<>();
    decision.put("aml_decision", out.getOrDefault("aml_decision","CLEAR"));
    decision.put("escalation_level", out.getOrDefault("escalation_level", 0));
    // 'flags' act as reasons
    Object flags = out.getOrDefault("flags", List.of());
    decision.put("reasons", flags);

    Map<String,Object> result = new LinkedHashMap<>();
    result.put("facts", facts);
    result.put("decision", decision);
    result.put("fired_rules", fired);

    System.out.println(M.writeValueAsString(result));
  }
}
