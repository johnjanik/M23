# Three-point class vectors of M23 that can carry a cover defined over Q, with
# normalised structure constants from the character table (GAP CTblLib).
#   gap -q scripts/gap/m23_passports.g < /dev/null | python3 -c "import sys,json; print(json.dumps(json.loads(sys.stdin.read().replace(chr(92)+chr(10),str())), indent=0))" > data/m23_three_point_passports.json
# Types: (R1,R2,R3) with rational classes at rational points, and (R, D, D') with
# D, D' = D^-1 an irrational pair at conjugate points of the quadratic field Q(chi_D).
# n = |C1||C2||C3| / |G|^2 * sum_chi chi(c1) chi(c2) chi(c3) / chi(1)
#   = number of triples (g1,g2,g3) in C1 x C2 x C3 with g1 g2 g3 = 1, divided by |G|.
# It equals the number of inner Nielsen classes when every such triple generates M23
# (true when a class of order 23 occurs); otherwise it is only an upper bound.
t := CharacterTable("M23");;
cn := ClassNames(t);; ord := OrdersClassRepresentatives(t);; sz := SizesConjugacyClasses(t);;
G := Size(t);; irr := Irr(t);;
perm := PermChars(t, 23)[1];;
# number of cycles on 23 points of an element of class i: (1/o) sum_k Fix(g^k)
cyc := function(i)
  local k, s, o; o := ord[i]; s := 0;
  for k in [1..o] do s := s + perm[PowerMap(t, k)[i]]; od;
  return s / o;
end;;
n3 := function(i, j, k)
  return sz[i]*sz[j]*sz[k]/G^2 * Sum(irr, x -> x[i]*x[j]*x[k]/x[1]);
end;;
rat := Filtered([2..Length(cn)], i -> ForAll(irr, x -> IsRat(x[i])));;
pairs := [];;
for i in [2..Length(cn)] do
  if not i in rat then
    j := PowerMap(t, -1)[i];
    if i < j then Add(pairs, [i, j]); fi;
  fi;
od;
rows := [];;
add := function(kind, i, j, k)
  local n, g2;
  n := n3(i, j, k);
  g2 := -46 + (23 - cyc(i)) + (23 - cyc(j)) + (23 - cyc(k));   # 2g - 2 of the degree 23 cover
  if n > 0 and g2 >= -2 then
    Add(rows, Concatenation("{\"type\":\"", kind, "\",\"classes\":[\"", cn[i], "\",\"", cn[j], "\",\"", cn[k],
        "\"],\"structure_constant\":\"", String(n), "\",\"genus23\":", String((g2+2)/2),
        ",\"exact_nielsen_count\":", String(23 in [ord[i], ord[j], ord[k]]), "}"));
  fi;
end;;
for a in [1..Length(rat)] do for b in [a..Length(rat)] do for c in [b..Length(rat)] do
  add("rational", rat[a], rat[b], rat[c]);
od; od; od;
for r in rat do for p in pairs do add("conjugate_pair", r, p[1], p[2]); od; od;
Print("{\"group\":\"M23\",\"source\":\"GAP CTblLib character table\",\"rational_classes\":[",
      JoinStringsWithSeparator(List(rat, i -> Concatenation("\"", cn[i], "\"")), ","), "],\"irrational_pairs\":[",
      JoinStringsWithSeparator(List(pairs, p -> Concatenation("[\"", cn[p[1]], "\",\"", cn[p[2]], "\"]")), ","),
      "],\n\"passports\":[\n", JoinStringsWithSeparator(rows, ",\n"), "\n]}\n");
