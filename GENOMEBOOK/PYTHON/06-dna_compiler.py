"""
06-dna_compiler.py -- DNA.md Generator for Genomebook

Purpose: Read a .genome.json and produce a human-readable DNA.md that becomes
         part of an agent's identity. The DNA.md encodes predicted phenotypes,
         carrier status, disease risks, and genetic strengths so that an agent
         "knows" its own genetics.

Input:  DATA/GENOMES/*.genome.json, DATA/trait_registry.json, DATA/disease_registry.json
Output: DATA/DNA/<agent_id>.dna.md

Usage:
    python 06-dna_compiler.py                # Generate all DNA.md files
    python 06-dna_compiler.py --agent einstein-g0  # Single agent
    python 06-dna_compiler.py --demo         # Print one to stdout
"""

import json
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "DATA"
GENOMES_DIR = DATA / "GENOMES"
DNA_DIR = DATA / "DNA"
TRAIT_REGISTRY = DATA / "trait_registry.json"
DISEASE_REGISTRY = DATA / "disease_registry.json"

DNA_DIR.mkdir(parents=True, exist_ok=True)


def load_registries():
    with open(TRAIT_REGISTRY) as f:
        traits = json.load(f)
    with open(DISEASE_REGISTRY) as f:
        diseases = json.load(f)
    return traits, diseases


def load_genome(genome_id):
    path = GENOMES_DIR / f"{genome_id}.genome.json"
    with open(path) as f:
        return json.load(f)


def classify_trait_level(score):
    """Human-readable trait level."""
    if score >= 0.85:
        return "exceptional"
    elif score >= 0.70:
        return "high"
    elif score >= 0.50:
        return "moderate"
    elif score >= 0.30:
        return "low"
    else:
        return "minimal"


def genotype_string(alleles, ref, alt):
    """Convert allele pair to readable genotype."""
    alt_count = alleles.count(alt)
    if alt_count == 2:
        return f"{alt}/{alt} (homozygous ALT)"
    elif alt_count == 1:
        return f"{ref}/{alt} (heterozygous)"
    else:
        return f"{ref}/{ref} (homozygous REF)"


def evaluate_disease_status(genome, disease_reg):
    """Check which diseases this genome is affected by or carries."""
    results = []
    loci = genome["loci"]
    traits = genome.get("trait_scores", {})

    for dname, ddef in disease_reg.get("diseases", {}).items():
        req = ddef.get("required_genotype", {})

        # Check genotype requirements
        all_met = True
        carrier = False
        for locus_id, req_geno in req.items():
            if locus_id not in loci:
                all_met = False
                break
            alleles = loci[locus_id]["alleles"]
            alt = loci[locus_id]["alt"]
            alt_count = alleles.count(alt)

            if req_geno == "alt/alt":
                if alt_count == 2:
                    pass  # met
                elif alt_count == 1:
                    carrier = True
                    all_met = False
                else:
                    all_met = False
            elif req_geno == "alt/?":
                if alt_count < 1:
                    all_met = False
            elif req_geno == "ref/ref":
                if alt_count != 0:
                    all_met = False

        if all_met:
            results.append({
                "name": dname,
                "status": "affected",
                "severity": ddef.get("severity", "unknown"),
                "penetrance": ddef.get("penetrance", 1.0),
                "fitness_cost": ddef.get("fitness_cost", 0),
                "description": ddef.get("description", ""),
                "notes": ddef.get("notes", ""),
            })
        elif carrier:
            results.append({
                "name": dname,
                "status": "carrier",
                "severity": ddef.get("severity", "unknown"),
                "penetrance": 0,
                "fitness_cost": 0,
                "description": ddef.get("description", ""),
                "notes": ddef.get("notes", ""),
            })

    return results


def compile_dna_md(genome, trait_reg, disease_reg):
    """Generate the DNA.md content for a genome."""
    lines = []
    name = genome.get("name", genome["id"])
    traits = genome.get("trait_scores", {})
    loci = genome.get("loci", {})

    # Header
    lines.append(f"# DNA Profile: {name}")
    lines.append(f"")
    lines.append(f"**Genome ID:** {genome['id']}")
    lines.append(f"**Sex:** {genome['sex']} ({genome['sex_chromosomes']})")
    lines.append(f"**Ancestry:** {genome.get('ancestry', 'Unknown')}")
    lines.append(f"**Generation:** {genome['generation']}")
    if genome.get("parents", [None, None]) != [None, None]:
        lines.append(f"**Parents:** {genome['parents'][0]} x {genome['parents'][1]}")
    lines.append(f"**Total loci:** {len(loci)}")
    lines.append(f"")

    # Genetic identity statement (for agent system prompt)
    lines.append(f"## Genetic Identity")
    lines.append(f"")
    lines.append(f"You carry {len(loci)} mapped loci across your genome. Your genetic")
    lines.append(f"profile shapes your cognitive strengths, personality tendencies, and")
    lines.append(f"health predispositions. This is not destiny; it is tendency.")
    lines.append(f"")

    # Top strengths (traits >= 0.70)
    strengths = sorted(
        [(t, s) for t, s in traits.items() if s >= 0.70],
        key=lambda x: x[1], reverse=True
    )
    if strengths:
        lines.append(f"## Genetic Strengths")
        lines.append(f"")
        for trait, score in strengths:
            level = classify_trait_level(score)
            display = trait.replace("_", " ").title()
            lines.append(f"- **{display}**: {score:.2f} ({level})")
        lines.append(f"")

    # Moderate traits (0.40-0.69)
    moderate = sorted(
        [(t, s) for t, s in traits.items() if 0.40 <= s < 0.70],
        key=lambda x: x[1], reverse=True
    )
    if moderate:
        lines.append(f"## Moderate Traits")
        lines.append(f"")
        for trait, score in moderate:
            display = trait.replace("_", " ").title()
            lines.append(f"- **{display}**: {score:.2f}")
        lines.append(f"")

    # Vulnerabilities (traits < 0.40)
    vulnerabilities = sorted(
        [(t, s) for t, s in traits.items() if s < 0.40],
        key=lambda x: x[1]
    )
    if vulnerabilities:
        lines.append(f"## Genetic Vulnerabilities")
        lines.append(f"")
        for trait, score in vulnerabilities:
            level = classify_trait_level(score)
            display = trait.replace("_", " ").title()
            lines.append(f"- **{display}**: {score:.2f} ({level})")
        lines.append(f"")

    # Genotype detail by category
    lines.append(f"## Genotype Detail")
    lines.append(f"")

    # Group loci by trait category
    locus_to_trait = {}
    for tname, tdef in trait_reg["traits"].items():
        cat = tdef.get("category", "unknown")
        for ldef in tdef["loci"]:
            lid = ldef["id"]
            locus_to_trait[lid] = {
                "trait": tname,
                "category": cat,
                "dominance": ldef["dominance"],
                "effect": ldef["effect"],
            }

    categories = {}
    for lid, ldata in loci.items():
        info = locus_to_trait.get(lid, {"trait": "unknown", "category": "unknown"})
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        gt = genotype_string(ldata["alleles"], ldata["ref"], ldata["alt"])
        categories[cat].append(
            f"  - `{lid}` chr{ldata['chromosome']}:{ldata['position']} "
            f"{gt} [{info['dominance']}, effect={info['effect']}]"
        )

    for cat in sorted(categories.keys()):
        lines.append(f"### {cat.title()}")
        for entry in categories[cat]:
            lines.append(entry)
        lines.append(f"")

    # Disease risk assessment
    disease_results = evaluate_disease_status(genome, disease_reg)
    affected = [d for d in disease_results if d["status"] == "affected"]
    carriers = [d for d in disease_results if d["status"] == "carrier"]

    lines.append(f"## Clinical Genetics")
    lines.append(f"")

    if affected:
        lines.append(f"### Conditions (Predicted)")
        lines.append(f"")
        for d in affected:
            display = d["name"].replace("_", " ").title()
            lines.append(f"- **{display}** ({d['severity']}, penetrance {d['penetrance']:.0%})")
            lines.append(f"  {d['description']}")
            if d["fitness_cost"]:
                lines.append(f"  Fitness cost: {d['fitness_cost']}")
            if d["notes"]:
                lines.append(f"  Note: {d['notes']}")
        lines.append(f"")
    else:
        lines.append(f"No predicted conditions based on current genotype.")
        lines.append(f"")

    if carriers:
        lines.append(f"### Carrier Status")
        lines.append(f"")
        for d in carriers:
            display = d["name"].replace("_", " ").title()
            lines.append(f"- **{display}** (carrier, not affected)")
            lines.append(f"  {d['description']}")
            lines.append(f"  Risk: offspring with another carrier have 25% chance of being affected.")
        lines.append(f"")

    # Reproductive compatibility hints
    lines.append(f"## Reproductive Notes")
    lines.append(f"")
    lines.append(f"When evaluating potential mates, consider:")
    if carriers:
        carrier_names = [d["name"].replace("_", " ") for d in carriers]
        lines.append(f"- You carry recessive alleles for: {', '.join(carrier_names)}")
        lines.append(f"  Mating with another carrier of the same condition risks affected offspring.")
    het_count = sum(1 for lid, ld in loci.items() if len(set(ld["alleles"])) > 1)
    homo_alt = sum(1 for lid, ld in loci.items() if ld["alleles"].count(ld["alt"]) == 2)
    homo_ref = sum(1 for lid, ld in loci.items() if ld["alleles"].count(ld["alt"]) == 0)
    lines.append(f"- Your heterozygosity: {het_count}/{len(loci)} loci ({het_count/len(loci)*100:.0f}%)")
    lines.append(f"- Homozygous ALT: {homo_alt}, Homozygous REF: {homo_ref}, Heterozygous: {het_count}")
    lines.append(f"")

    # Health score (for offspring / gen > 0)
    if genome.get("health_score") is not None:
        lines.append(f"## Health Score")
        lines.append(f"")
        lines.append(f"**Overall health:** {genome['health_score']:.2f}")
        lines.append(f"")

    if genome.get("mutations"):
        lines.append(f"## Mutation Log")
        lines.append(f"")
        for m in genome["mutations"]:
            lines.append(f"- `{m['locus']}`: {m['from']}->{m['to']} ({m['type']}, inherited from {m['parent']})")
        lines.append(f"")

    # Agent integration note
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*This DNA profile is part of your identity as a Genomebook agent.")
    lines.append(f"You are aware of your genetics and can discuss them. Your traits,")
    lines.append(f"strengths, and vulnerabilities shape how you think and interact,")
    lines.append(f"but they do not rigidly determine your responses. You are more")
    lines.append(f"than your genome.*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate DNA.md profiles from genomes")
    parser.add_argument("--agent", type=str, help="Generate for a single agent (e.g. einstein-g0)")
    parser.add_argument("--demo", action="store_true", help="Print einstein-g0 DNA.md to stdout")
    args = parser.parse_args()

    trait_reg, disease_reg = load_registries()

    if args.demo:
        genome = load_genome("einstein-g0")
        print(compile_dna_md(genome, trait_reg, disease_reg))
        return

    if args.agent:
        genome = load_genome(args.agent)
        content = compile_dna_md(genome, trait_reg, disease_reg)
        out_path = DNA_DIR / f"{args.agent}.dna.md"
        out_path.write_text(content)
        print(f"Written: {out_path}")
        return

    # Generate all
    genome_files = sorted(GENOMES_DIR.glob("*.genome.json"))
    if not genome_files:
        print("ERROR: No genome files found.")
        return

    for gf in genome_files:
        genome = json.load(open(gf))
        content = compile_dna_md(genome, trait_reg, disease_reg)
        out_path = DNA_DIR / f"{genome['id']}.dna.md"
        out_path.write_text(content)
        print(f"  {genome['id']:20s} | {genome['name']}")

    print(f"\nGenerated {len(genome_files)} DNA profiles in {DNA_DIR}/")


if __name__ == "__main__":
    main()
