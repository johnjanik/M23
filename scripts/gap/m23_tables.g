# Cross-checks with the GAP libraries CTblLib and TomLib (not needed for the Python scripts).
#   gap -q scripts/gap/m23_tables.g < /dev/null
t := CharacterTable("M23");
tom := TableOfMarks("M23");
pcs := PermCharsTom(t, tom);
Print("classes of subgroups: ", Length(pcs), "\n");
Print("index 253 permutation characters, as multiplicities of irreducibles:\n",
      List(Filtered(pcs, x -> x[1] = 253), x -> MatScalarProducts(t, Irr(t), [x])[1]), "\n");
pairs := [];
for i in [1..Length(pcs)] do for j in [i+1..Length(pcs)] do
  if pcs[i] = pcs[j] then Add(pairs, pcs[i][1]); fi;
od; od;
Sort(pairs);
Print("Gassmann-equivalent pairs of classes: ", Length(pairs), "; smallest indices: ", pairs{[1..6]}, "\n");
cn := ClassNames(t);
Print("characters irrational on elements of order 23: degrees ",
      List(Filtered(Irr(t), x -> not IsRat(x[Position(cn, "23a")])), x -> x[1]), "\n");
tm := t mod 2;
Print("2-modular irreducible degrees: ", List(Irr(tm), x -> x[1]), "\n");
Print("decomposition of the degree 22 character mod 2: ", DecompositionMatrix(tm)[2], "\n");
