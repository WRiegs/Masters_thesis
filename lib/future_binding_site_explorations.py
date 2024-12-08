import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import subprocess
from Bio import SeqIO

# Cofactor Chebi ontology numbers for all cofactors

#chebi_numbers = [
#    17579, 23357, 15636, 1989, 57453, 15635, 83088, 30314, 16509, 80214, 
#    28889, 17679, 60494, 17601, 58201, 52020, 40260, 60470, 17401, 58130, 
#    16680, 15414, 59789, 38290, 29073, 16027, 456215, 28938, 15422, 57299, 
#    30616, 61338, 37136, 15956, 57586, 30402, 48775, 29108, 48782, 17996, 
#    18230, 58416, 27888, 61721, 15982, 60488, 16304, 30411, 48828, 49415, 
#    18408, 28265, 60540, 23378, 49552, 29036, 33221, 33913, 72953, 17803, 
#    35169, 17694, 42121, 60342, 73113, 73095, 73115, 73096, 4746, 16238, 
#    57692, 60454, 68438, 60519, 36183, 60532, 17627, 60344, 61717, 60562, 
#    24040, 24041, 17621, 58210, 16048, 57618, 57925, 16856, 62811, 62814, 
#    24480, 16240, 17544, 43474, 17594, 24875, 29033, 29034, 60504, 30409, 
#    60357, 49701, 49807, 18420, 29035, 44245, 28115, 25372, 16768, 57540, 
#    15846, 16908, 57945, 58349, 18009, 16474, 57783, 137373, 137399, 25516, 
#    49786, 17154, 47739, 177874, 16858, 47942, 25848, 18067, 26116, 36079, 
#    26214, 29103, 150862, 87749, 87746, 87531, 17310, 18405, 597326, 131529, 
#    16709, 26461, 18315, 58442, 15361, 32816, 17015, 57986, 59560, 28599, 
#    60052, 18023, 58351, 29101, 35104, 16189, 49883, 71177, 9532, 58937, 
#    35172, 36970, 27547, 29105, 23354, 26348, 26672
#]

chebi_numbers = """β-carotene (CHEBI:17579) has role cofactor (CHEBI:23357)
(6R)-5,10-methylenetetrahydrofolate(2−) (CHEBI:15636) has role cofactor (CHEBI:23357)
(6R)-5,10-methylenetetrahydrofolic acid (CHEBI:1989) has role cofactor (CHEBI:23357)
(6S)-5,6,7,8-tetrahydrofolate(2−) (CHEBI:57453) has role cofactor (CHEBI:23357)
(6S)-5,6,7,8-tetrahydrofolic acid (CHEBI:15635) has role cofactor (CHEBI:23357)
(R)-lipoate (CHEBI:83088) has role cofactor (CHEBI:23357)
(R)-lipoic acid (CHEBI:30314) has role cofactor (CHEBI:23357)
1,4-benzoquinone (CHEBI:16509) has role cofactor (CHEBI:23357)
3'-hydroxyechinenone (CHEBI:80214) has role cofactor (CHEBI:23357)
5,6,7,8-tetrahydropteridine (CHEBI:28889) has role cofactor (CHEBI:23357)
5-hydroxybenzimidazolylcob(I)amide (CHEBI:17679) has role cofactor (CHEBI:23357)
5-hydroxybenzimidazolylcob(I)amide(1−) (CHEBI:60494) has role cofactor (CHEBI:23357)
6,7-dimethyl-8-(1-D-ribityl)lumazine (CHEBI:17601) has role cofactor (CHEBI:23357)
6,7-dimethyl-8-(1-D-ribityl)lumazine(1−) (CHEBI:58201) has role cofactor (CHEBI:23357)
6-decylubiquinone (CHEBI:52020) has role cofactor (CHEBI:23357)
6-hydroxy-FAD (CHEBI:40260) has role cofactor (CHEBI:23357)
6-hydroxy-FAD(3−) (CHEBI:60470) has role cofactor (CHEBI:23357)
myo-inositol hexakisphosphate (CHEBI:17401) has role cofactor (CHEBI:23357)
myo-inositol hexakisphosphate(12−) (CHEBI:58130) has role cofactor (CHEBI:23357)
S-adenosyl-L-homocysteine (CHEBI:16680) has role cofactor (CHEBI:23357)
S-adenosyl-L-methionine (CHEBI:15414) has role cofactor (CHEBI:23357)
S-adenosyl-L-methionine zwitterion (CHEBI:59789) has role cofactor (CHEBI:23357)
L-ascorbate (CHEBI:38290) has role cofactor (CHEBI:23357)
L-ascorbic acid (CHEBI:29073) has role cofactor (CHEBI:23357)
adenosine 5'-monophosphate (CHEBI:16027) has role cofactor (CHEBI:23357)
adenosine 5'-monophosphate(2−) (CHEBI:456215) has role cofactor (CHEBI:23357)
ammonium (CHEBI:28938) has role cofactor (CHEBI:23357)
ATP (CHEBI:15422) has role cofactor (CHEBI:23357)
ATP(3−) (CHEBI:57299) has role cofactor (CHEBI:23357)
ATP(4−) (CHEBI:30616) has role cofactor (CHEBI:23357)
bacillithiol (CHEBI:61338) has role cofactor (CHEBI:23357)
barium(2+) (CHEBI:37136) has role cofactor (CHEBI:23357)
biotin (CHEBI:15956) has role cofactor (CHEBI:23357)
biotinate (CHEBI:57586) has role cofactor (CHEBI:23357)
bis(molybdopterin)tungsten cofactor (CHEBI:30402) has role cofactor (CHEBI:23357)
cadmium(2+) (CHEBI:48775) has role cofactor (CHEBI:23357)
calcium(2+) (CHEBI:29108) has role cofactor (CHEBI:23357)
cerium(3+) (CHEBI:48782) has role cofactor (CHEBI:23357)
chloride (CHEBI:17996) has role cofactor (CHEBI:23357)
chlorophyll a (CHEBI:18230) has role cofactor (CHEBI:23357)
chlorophyll a(1−) (CHEBI:58416) has role cofactor (CHEBI:23357)
chlorophyll b (CHEBI:27888) has role cofactor (CHEBI:23357)
chlorophyll b(1−) (CHEBI:61721) has role cofactor (CHEBI:23357)
cob(I)alamin (CHEBI:15982) has role cofactor (CHEBI:23357)
cob(I)alamin(1−) (CHEBI:60488) has role cofactor (CHEBI:23357)
cob(II)alamin (CHEBI:16304) has role cofactor (CHEBI:23357)
cobalamin (CHEBI:30411) has role cofactor (CHEBI:23357)
cobalt(2+) (CHEBI:48828) has role cofactor (CHEBI:23357)
cobalt(3+) (CHEBI:49415) has role cofactor (CHEBI:23357)
cobamamide (CHEBI:18408) has role cofactor (CHEBI:23357)
coenzyme F430 (CHEBI:28265) has role cofactor (CHEBI:23357)
coenzyme F430(5−) (CHEBI:60540) has role cofactor (CHEBI:23357)
copper cation (CHEBI:23378) has role cofactor (CHEBI:23357)
copper(1+) (CHEBI:49552) has role cofactor (CHEBI:23357)
copper(2+) (CHEBI:29036) has role cofactor (CHEBI:23357)
corrin (CHEBI:33221) has role cofactor (CHEBI:23357)
corrinoid (CHEBI:33913) has role cofactor (CHEBI:23357)
decylplastoquinone (CHEBI:72953) has role cofactor (CHEBI:23357)
dehydro-D-arabinono-1,4-lactone (CHEBI:17803) has role cofactor (CHEBI:23357)
dihydrogenvanadate (CHEBI:35169) has role cofactor (CHEBI:23357)
dihydrolipoamide (CHEBI:17694) has role cofactor (CHEBI:23357)
dipyrromethane cofactor (CHEBI:42121) has role cofactor (CHEBI:23357)
dipyrromethane cofactor(4−) (CHEBI:60342) has role cofactor (CHEBI:23357)
divinyl chlorophyll a (CHEBI:73113) has role cofactor (CHEBI:23357)
divinyl chlorophyll a(1−) (CHEBI:73095) has role cofactor (CHEBI:23357)
divinyl chlorophyll b (CHEBI:73115) has role cofactor (CHEBI:23357)
divinyl chlorophyll b(1−) (CHEBI:73096) has role cofactor (CHEBI:23357)
echinenone (CHEBI:4746) has role cofactor (CHEBI:23357)
FAD (CHEBI:16238) has role cofactor (CHEBI:23357)
FAD(3−) (CHEBI:57692) has role cofactor (CHEBI:23357)
Fe-coproporphyrin III (CHEBI:60454) has role cofactor (CHEBI:23357)
Fe-coproporphyrin III(4−) (CHEBI:68438) has role cofactor (CHEBI:23357)
Fe4S2O2 iron-sulfur-oxygen cluster (CHEBI:60519) has role cofactor (CHEBI:23357)
ferriheme a (CHEBI:36183) has role cofactor (CHEBI:23357)
ferriheme a(1−) (CHEBI:60532) has role cofactor (CHEBI:23357)
ferroheme b (CHEBI:17627) has role cofactor (CHEBI:23357)
ferroheme b(2−) (CHEBI:60344) has role cofactor (CHEBI:23357)
ferroheme c(2−) (CHEBI:61717) has role cofactor (CHEBI:23357)
ferroheme c (CHEBI:60562) has role cofactor (CHEBI:23357)
flavin adenine dinucleotide (CHEBI:24040) has role cofactor (CHEBI:23357)
flavin mononucleotide (CHEBI:24041) has role cofactor (CHEBI:23357)
FMN (CHEBI:17621) has role cofactor (CHEBI:23357)
FMN(3−) (CHEBI:58210) has role cofactor (CHEBI:23357)
FMNH2 (CHEBI:16048) has role cofactor (CHEBI:23357)
FMNH2(2−) (CHEBI:57618) has role cofactor (CHEBI:23357)
glutathionate(1−) (CHEBI:57925) has role cofactor (CHEBI:23357)
glutathione (CHEBI:16856) has role cofactor (CHEBI:23357)
heme d cis-diol (CHEBI:62811) has role cofactor (CHEBI:23357)
heme d cis-diol(2−) (CHEBI:62814) has role cofactor (CHEBI:23357)
heme o (CHEBI:24480) has role cofactor (CHEBI:23357)
hydrogen peroxide (CHEBI:16240) has role cofactor (CHEBI:23357)
hydrogencarbonate (CHEBI:17544) has role cofactor (CHEBI:23357)
hydrogenphosphate (CHEBI:43474) has role cofactor (CHEBI:23357)
hydroquinone (CHEBI:17594) has role cofactor (CHEBI:23357)
iron cation (CHEBI:24875) has role cofactor (CHEBI:23357)
iron(2+) (CHEBI:29033) has role cofactor (CHEBI:23357)
iron(3+) (CHEBI:29034) has role cofactor (CHEBI:23357)
iron-sulfur-iron cofactor (CHEBI:60504) has role cofactor (CHEBI:23357)
iron-sulfur-molybdenum cofactor (CHEBI:30409) has role cofactor (CHEBI:23357)
iron-sulfur-vanadium cofactor (CHEBI:60357) has role cofactor (CHEBI:23357)
lanthanum cation (CHEBI:231841) has role cofactor (CHEBI:23357)
lanthanum(3+) (CHEBI:49701) has role cofactor (CHEBI:23357)
lead(2+) (CHEBI:49807) has role cofactor (CHEBI:23357)
magnesium(2+) (CHEBI:18420) has role cofactor (CHEBI:23357)
manganese(2+) (CHEBI:29035) has role cofactor (CHEBI:23357)
menaquinone-7 (CHEBI:44245) has role cofactor (CHEBI:23357)
methylcobalamin (CHEBI:28115) has role cofactor (CHEBI:23357)
molybdopterin cofactor (CHEBI:25372) has role cofactor (CHEBI:23357)
mycothiol (CHEBI:16768) has role cofactor (CHEBI:23357)
NAD(1−) (CHEBI:57540) has role cofactor (CHEBI:23357)
NAD+ (CHEBI:15846) has role cofactor (CHEBI:23357)
NADH (CHEBI:16908) has role cofactor (CHEBI:23357)
NADH(2−) (CHEBI:57945) has role cofactor (CHEBI:23357)
NADP(3−) (CHEBI:58349) has role cofactor (CHEBI:23357)
NADP+ (CHEBI:18009) has role cofactor (CHEBI:23357)
NADPH (CHEBI:16474) has role cofactor (CHEBI:23357)
NADPH(4−) (CHEBI:57783) has role cofactor (CHEBI:23357)
Ni(II)-pyridinium-3,5-bisthiocarboxylate mononucleotide(1−) (CHEBI:137373) has role cofactor (CHEBI:23357)
Ni(II)-pyridinium-3,5-bisthiocarboxylic acid mononucleotide (CHEBI:137399) has role cofactor (CHEBI:23357)
nickel cation (CHEBI:25516) has role cofactor (CHEBI:23357)
nickel(2+) (CHEBI:49786) has role cofactor (CHEBI:23357)
nicotinamide (CHEBI:17154) has role cofactor (CHEBI:23357)
NiFe4S4 cluster (CHEBI:47739) has role cofactor (CHEBI:23357)
NiFe4S5 cluster (CHEBI:177874) has role cofactor (CHEBI:23357)
pantetheine 4'-phosphate (CHEBI:16858) has role cofactor (CHEBI:23357)
pantetheine 4'-phosphate(2−) (CHEBI:47942) has role cofactor (CHEBI:23357)
pantothenic acids (CHEBI:25848) has role cofactor (CHEBI:23357)
phylloquinone (CHEBI:18067) has role cofactor (CHEBI:23357)
phytochromobilin (CHEBI:26116) has role cofactor (CHEBI:23357)
polypeptide-derived cofactor (CHEBI:36079) has role cofactor (CHEBI:23357)
porphyrins (CHEBI:26214) has role cofactor (CHEBI:23357)
potassium(1+) (CHEBI:29103) has role cofactor (CHEBI:23357)
premycofactocin (CHEBI:150862) has role cofactor (CHEBI:23357)
prenyl-FMN (CHEBI:87749) has role cofactor (CHEBI:23357)
prenyl-FMN(2−) (CHEBI:87746) has role cofactor (CHEBI:23357)
prenyl-FMNH2 (CHEBI:87531) has role cofactor (CHEBI:23357)
pyridoxal (CHEBI:17310) has role cofactor (CHEBI:23357)
pyridoxal 5'-phosphate (CHEBI:18405) has role cofactor (CHEBI:23357)
pyridoxal 5'-phosphate(2−) (CHEBI:597326) has role cofactor (CHEBI:23357)
pyridoxal hydrochloride (CHEBI:131529) has role cofactor (CHEBI:23357)
pyridoxine (CHEBI:16709) has role cofactor (CHEBI:23357)
pyrroloquinoline cofactor (CHEBI:26461) has role cofactor (CHEBI:23357)
pyrroloquinoline quinone (CHEBI:18315) has role cofactor (CHEBI:23357)
pyrroloquinoline quinone(3−) (CHEBI:58442) has role cofactor (CHEBI:23357)
pyruvate (CHEBI:15361) has role cofactor (CHEBI:23357)
pyruvic acid (CHEBI:32816) has role cofactor (CHEBI:23357)
riboflavin (CHEBI:17015) has role cofactor (CHEBI:23357)
riboflavin(1−) (CHEBI:57986) has role cofactor (CHEBI:23357)
sapropterin (CHEBI:59560) has role cofactor (CHEBI:23357)
siroheme (CHEBI:28599) has role cofactor (CHEBI:23357)
siroheme(8−) (CHEBI:60052) has role cofactor (CHEBI:23357)
sirohydrochlorin (CHEBI:18023) has role cofactor (CHEBI:23357)
sirohydrochlorin(8−) (CHEBI:58351) has role cofactor (CHEBI:23357)
sodium(1+) (CHEBI:29101) has role cofactor (CHEBI:23357)
strontium(2+) (CHEBI:35104) has role cofactor (CHEBI:23357)
sulfate (CHEBI:16189) has role cofactor (CHEBI:23357)
tetra-μ3-sulfido-tetrairon (CHEBI:49883) has role cofactor (CHEBI:23357)
tetrahydromonapterin (CHEBI:71177) has role cofactor (CHEBI:23357)
thiamine(1+) diphosphate (CHEBI:9532) has role cofactor (CHEBI:23357)
thiamine(1+) diphosphate(3−) (CHEBI:58937) has role cofactor (CHEBI:23357)
vanadium cation (CHEBI:35172) has role cofactor (CHEBI:23357)
vitamin B6 phosphate (CHEBI:36970) has role cofactor (CHEBI:23357)
zeaxanthin (CHEBI:27547) has role cofactor (CHEBI:23357)
zinc(2+) (CHEBI:29105) has role cofactor (CHEBI:23357)
coenzyme (CHEBI:23354) is a cofactor (CHEBI:23357)
prosthetic group (CHEBI:26348) is a cofactor (CHEBI:23357)
siderophore (CHEBI:26672) is a cofactor (CHEBI:23357)"""

chebi_numbers = chebi_numbers.split("(CHEBI:")
# drop the 0th element
chebi_numbers = chebi_numbers[1:]
chebi_numbers = [int(x.split(")")[0]) for x in chebi_numbers]
chebi_numbers = set(chebi_numbers)

# https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI%3A23357


def load_fasta(fasta_file):
    seqs = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seqs.append(record)
    return seqs


def load_cdhit(clstr_file):
    clusters = []
    with open(clstr_file) as f:
        cluster = []
        for line in f:
            if line.startswith(">"):
                if cluster:
                    clusters.append(cluster)
                cluster = []
            else:
                cluster.append(line.strip())
        clusters.append(cluster)
    return clusters


def filter_low_identity_sequences(input_fasta, output_fasta, similarity_threshold=0.3):
    # Define temporary directories
    tmp_dir = "mmseqs_tmp"
    db_dir = os.path.join(tmp_dir, "db")
    clustered_dir = os.path.join(tmp_dir, "clustered")
    repseqs_fasta = os.path.join(tmp_dir, "repseqs")
    
    # Create temporary directories
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        # Convert FASTA to MMseqs2 database
        cmd_convert = [
            "mmseqs", "createdb", input_fasta, db_dir
        ]
        subprocess.run(cmd_convert, check=True)

        # Perform clustering with the specified similarity threshold
        cmd_cluster = [
            "mmseqs", "cluster", db_dir, clustered_dir, tmp_dir,
            "--min-seq-id", str(similarity_threshold),
            "-c", "0.8", # Coverage threshold
            "--cov-mode", "0", # Use the full alignment for coverage calculation
            "--threads", "4" # Number of threads to use
        ]
        subprocess.run(cmd_cluster, check=True)

        # Get representative sequences (these will be the representatives of the clusters)
        cmd_repseq = [
            "mmseqs", "result2repseq", db_dir, clustered_dir, repseqs_fasta
        ]
        subprocess.run(cmd_repseq, check=True)

        # Extract sequences that were not part of any cluster (low identity sequences)
        cmd_extract = [
            "mmseqs", "createsubdb", repseqs_fasta, db_dir, output_fasta
        ]
        subprocess.run(cmd_extract, check=True)

        # Convert the final database back to a FASTA file
        cmd_export = [
            "mmseqs", "convert2fasta", output_fasta, output_fasta + ".fasta"
        ]
        subprocess.run(cmd_export, check=True)

        # Parse the resulting FASTA to get a list of sequence IDs
        low_identity_sequences = []
        with open(output_fasta + ".fasta", "r") as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                low_identity_sequences.append(record.id)

    finally:
        # Clean up temporary directories
        if os.path.exists(tmp_dir):
            subprocess.run(["rm", "-r", tmp_dir])

    return low_identity_sequences


def get_cluster_seqs(cluster):
    redundant_seqs = []
    for seq in cluster:
        redundant_seqs.append(seq.split(">")[1].split("|")[1])
    return redundant_seqs


def run_cd_hit(fasta_path, output_path, threshold=0.9):
    os.system(f'cd-hit -i {fasta_path} -o {output_path}/{threshold}clusters.fasta -c {threshold} -T 4 -M 2000')
    clusters = load_cdhit(f'{output_path}/{threshold}clusters.fasta.clstr')
    clusters = [cluster for cluster in clusters if len(cluster) > 1]
    print(f'Number of clusters: {len(clusters)}')
    
    # Go through the clusters, and get the >40% seqid sequences
    high_identity = []
    for i in clusters:
        seqs = get_cluster_seqs(i)
        for seq in seqs:
            if seq not in high_identity:
                high_identity.append(seq)
            else:
                if seq in high_identity:
                    print("Bro we deleted him! :O")
                    
    # get all seqs in the fasta file that are not in high_identity
    fasta = fasta_path
    seqs= load_fasta(fasta)
    low_identity = []

    for seq in seqs:
        id = seq.id.split("|")[1]
        if id not in high_identity:
            low_identity.append(id)

    print(f'<{threshold} id seqs: ',len(low_identity))
    print('total seqs: ',len(seqs))

    # Create a new fasta file with the non-redundant sequences
    with open(f'{output_path}/{threshold}filtered_for_redundancy.fasta', "w") as f:
        for seq in seqs:
            id = seq.id.split("|")[1]
            if id in low_identity:
                f.write(f">{seq.id}\n{seq.seq}\n")
                
    return high_identity, low_identity


def get_cofactor_sites(df):
    # remove any binding that has /ligand_id=ChEBI:XXXXX/ where XXXXX not in chebi_numbers
    cofactor_sites_per_seq = []
    cofactor_IDs_per_seq = []
    for i in range(len(df)):
        cofactor_sites = []
        cofactor_IDs = []
        binding = df['Binding site'][i]
        # if nan, skip
        if pd.isna(binding):
            cofactor_sites_per_seq.append(None)
            cofactor_IDs_per_seq.append(None)
            continue
        binding = binding.replace('\"', '')
        binding = binding.replace(' ', '')
        binding = binding.replace('\'', '')
        binding = binding.split(";")
        for j in binding:
            if j.startswith("BINDING"):
                sites = (j.split('BINDING')[1]).strip()
                sites = sites.split('..')
                if len(sites) == 2:
                    try:
                        sites = np.arange(int(sites[0])-1,int(sites[1]))
                        found = True
                    except:
                        found = False
                else:
                    try:
                        sites = [int(sites[0])-1]
                        found = True
                    except:
                        found = False
            if found is False:
                continue
            if j.startswith("/ligand_id=ChEBI:CHEBI:") or j.startswith("/ligand_id=ChEBI:"): 
                j = j.replace("ChEBI:", "")
                j = j.replace("CHEBI:", "")
                chebi_id = int(j.split("=")[1])
                if chebi_id not in chebi_numbers:
                    found = False
                    continue
                cofactor_sites.extend(sites)
                cofactor_IDs.extend([chebi_id]*len(sites))
                found = False
            # look for substrate binding sites
            if j.startswith("/ligand=substrate") or j.startswith("/ligand=Substrate"):
                cofactor_sites.extend(sites)
                cofactor_IDs.extend([00000]*len(sites))
                found = False
        if len(cofactor_sites) == 0:
            cofactor_sites_per_seq.append(None)
            cofactor_IDs_per_seq.append(None)
        else:
            cofactor_sites_per_seq.append(cofactor_sites)
            cofactor_IDs_per_seq.append(cofactor_IDs)
    
    assert len(cofactor_sites_per_seq) == len(df), "Length of cofactor sites and master_df do not match"
    assert len(cofactor_IDs_per_seq) == len(df), "Length of cofactor IDs and master_df do not match"
    
    return cofactor_sites_per_seq, cofactor_IDs_per_seq


def get_EC_tiers(df):
    # for both df, get the EC number for each tier, [0:4] and make a column for each
    ECs = [str(x) for x in df['EC number']]
    ECs = [x.replace(' ','') for x in ECs]
    ECs = [x.split(';') for x in ECs]
    ECs = [[x.split('.') for x in y] for y in ECs]
    ECs_tier1 = [['.'.join(x[0:1]) for x in y] for y in ECs]
    ECs_tier2 = [['.'.join(x[0:2]) for x in y] for y in ECs]
    ECs_tier3 = [['.'.join(x[0:3]) for x in y] for y in ECs]
    ECs_tier4 = [['.'.join(x[0:4]) for x in y] for y in ECs]
    df['EC_tier1'] = ECs_tier1
    df['EC_tier2'] = ECs_tier2
    df['EC_tier3'] = ECs_tier3
    df['EC_tier4'] = ECs_tier4
    return df


def get_val_counts_EC_tiers(df):
    # account for sublists in the EC_tier columns
    EC_tier1 = []
    EC_tier2 = []
    EC_tier3 = []
    EC_tier4 = []
    for i in range(len(df)):
        EC_tier1.extend(df['EC_tier1'][i])
        EC_tier2.extend(df['EC_tier2'][i])
        EC_tier3.extend(df['EC_tier3'][i])
        EC_tier4.extend(df['EC_tier4'][i])
    EC_tier1 = pd.Series(EC_tier1).value_counts()
    EC_tier2 = pd.Series(EC_tier2).value_counts()
    EC_tier3 = pd.Series(EC_tier3).value_counts()
    EC_tier4 = pd.Series(EC_tier4).value_counts()
    return EC_tier1, EC_tier2, EC_tier3, EC_tier4


def plot_2_distributions(vc1, vc2, out_path):
    # normalise the value counts so they are comparable
    vc1 = vc1/vc1.sum()
    vc2 = vc2/vc2.sum()
    fig, ax = plt.subplots(figsize=(20, 12))
    # keep the bars next to each other, not overlapping
    width = 0.35
    x = np.arange(len(vc1))
    ax.bar(x - width/2, vc1, width, label='Active site')
    ax.bar(x + width/2, vc2, width, label='Binding site')
    ax.set_xticks(x)
    ax.set_xticklabels(vc1.index)
    ax.legend()
    plt.xticks(rotation=45)
    plt.savefig(out_path)
    
    

def plot_4_distributions(vc1, vc2, vc3, vc4, out_path):
    # normalise the value counts so they are comparable
    vc1 = vc1/vc1.sum()
    vc2 = vc2/vc2.sum()
    vc3 = vc3/vc3.sum()
    vc4 = vc4/vc4.sum()

    fig, ax = plt.subplots(figsize=(40, 24))
    # keep the bars next to each other, not overlapping
    width = 0.2
    x = np.arange(len(vc1))
    ax.bar(x - width, vc1, width, label='Tier 1')
    ax.bar(x, vc2, width, label='Tier 2')
    ax.bar(x + width, vc3, width, label='Tier 3')
    ax.bar(x + 2*width, vc4, width, label='Tier 4')
    ax.set_xticks(x)
    ax.set_xticklabels(vc1.index)
    ax.legend()
    plt.xticks(rotation=45)
    plt.savefig(out_path)
    

def plot_4_distributions(vc1, vc2, vc3, vc4, out_path):
    # normalise the value counts so they are comparable
    fig, ax = plt.subplots(figsize=(20, 12))
    # keep the bars next to each other, not overlapping
    width = 0.2
    x = np.arange(max(len(vc1), len(vc2), len(vc3), len(vc4)))
    ax.bar(x - width, vc1, width, label='All with AS')
    ax.bar(x, vc2, width, label='All with BS')
    ax.bar(x + width, vc3, width, label='BS but no AS')
    ax.bar(x + 2*width, vc4, width, label='AS but no BS')
    ax.set_xticks(x)
    ax.set_xticklabels(vc1.index)
    ax.legend()
    plt.xticks(rotation=45)
    plt.savefig(out_path)
    
    

def plot_2_distributions(vc1, vc2, out_path):
    # if label not in vc1, or vice versa, add it with a value of 0
    for i in vc1.index:
        if i not in vc2.index:
            vc2[i] = 0
    for i in vc2.index:
        if i not in vc1.index:
            vc1[i] = 0
    
    # normalise the value counts so they are comparable
    fig, ax = plt.subplots(figsize=(20, 12))
    # keep the bars next to each other, not overlapping
    width = 0.35
    x = np.arange(max(len(vc1), len(vc2)))
    ax.bar(x - width/2, vc1, width, label='Active site')
    ax.bar(x + width/2, vc2, width, label='Binding site')
    ax.set_xticks(x)
    ax.set_xticklabels(vc1.index)
    ax.legend()
    plt.xticks(rotation=45)
    plt.savefig(out_path)


def main():
    df_path = "/scratch/project/squid/OMEGA/EC_ASorBS_1024.tsv"
    #df_path = "/scratch/project/squid/OMEGA/BS_model/metadata_paired.tsv"
    out_path = "/scratch/project/squid/OMEGA/visualising_bias/"

    master_df = pd.read_csv(df_path, sep="\t")

    # get the cofactor sites and IDs
    cofactor_sites_per_seq, cofactor_IDs_per_seq = get_cofactor_sites(master_df)
    master_df['Cofactor sites'] = cofactor_sites_per_seq
    master_df['Cofactor IDs'] = cofactor_IDs_per_seq
    
    print(master_df['Cofactor sites'])
    print(master_df['Cofactor IDs'])
    
    # show value counts of cofactor IDs
    cofactor_IDs = []
    for i in range(len(master_df)):
        if master_df['Cofactor IDs'][i] == None:
            cofactor_IDs.append(None)
        else:
            cofactor_IDs.extend(master_df['Cofactor IDs'][i])
    cofactor_IDs = pd.Series(cofactor_IDs).value_counts()
    
    # get the list of cofactor IDs that are below a threshold that makes them uncommon:
    uncommon = cofactor_IDs[cofactor_IDs < 70]
    print("Number of cofactor IDs:")
    print(len(cofactor_IDs))
    print(cofactor_IDs)
    
    print("Number of uncommon cofactors:")
    print(len(uncommon))
    print(uncommon)
    
    # CHEBI:35310 Carotenones
    # If has part or is CHEBI:48796 iron sulfur cluster
    # cyclic tetrapyrrole anion (CHEBI:58941) 
    #  is a nickel-iron-sulfur cluster (CHEBI:60400)
    # is a cobalamins (CHEBI:23334) - huge mfs
    # dihydrogenvanadate (CHEBI:35169) is a monovalent inorganic anion (CHEBI:79389) dihydrogenvanadate (CHEBI:35169) is a vanadium oxoanion (CHEBI:30528) 
    # corrinoid (CHEBI:33913) is a cyclic tetrapyrrole (CHEBI:36309)
    # coenzyme F430(5−) (CHEBI:60540) has role cofactor (CHEBI:23357) coenzyme F430(5−) (CHEBI:60540) is a pentacarboxylic acid anion (CHEBI:35755) coenzyme F430(5−) (CHEBI:60540) is conjugate base of coenzyme F430 (CHEBI:28265) 
    
    # get the number of occurances of each cofactor ID in the dataset, but only count 1 per sequence, not per site!
    cofactor_IDs_per_seq = []
    for i in range(len(master_df)):
        if master_df['Cofactor IDs'][i] == None:
            cofactor_IDs_per_seq.append(None)
        else:
            cofactor_IDs_per_seq.extend(set(master_df['Cofactor IDs'][i]))
    cofactor_IDs_per_seq = pd.Series(cofactor_IDs_per_seq).value_counts()
    print("Number of cofactor IDs per sequence:")
    print(len(cofactor_IDs_per_seq))
    print(cofactor_IDs_per_seq)
    uncommon = cofactor_IDs[cofactor_IDs < 50]
    
    print("Number of uncommon cofactors per sequence:")
    print(len(uncommon))
    print(uncommon)
    
    
    
    # plot the distribution of cofactor IDs using a bar chart from seaborn
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(x=cofactor_IDs.index, y=cofactor_IDs.values, ax=ax)
    plt.xticks(rotation=45)
    plt.savefig(out_path + "cofactor_IDs.png")
    
    # get how many cofactor sites have less than 30 instances
    cofactor_sites = []

    # remove any rows with length >1024
    master_df = master_df[master_df['Length'] <= 1024]    
    
    # remove any rows with no EC number
    master_df = master_df.dropna(subset=['EC number'])
        
    # reset the index
    master_df = master_df.reset_index(drop=True)
    
    master_df = get_EC_tiers(master_df)
    
    has_AS_df = master_df[master_df['Active site'].notna()]
    has_AS_df = has_AS_df.reset_index(drop=True)
    has_AS_no_BS_df = has_AS_df[has_AS_df['Cofactor sites'].isna()]
    has_AS_no_BS_df = has_AS_no_BS_df.reset_index(drop=True)
    has_BS_df = master_df[master_df['Cofactor sites'].notna()]
    has_BS_df = has_BS_df.reset_index(drop=True)
    has_BS_no_AS_df = has_BS_df[has_BS_df['Active site'].isna()]
    has_BS_no_AS_df = has_BS_no_AS_df.reset_index(drop=True)
    
    EC_tier1, EC_tier2, EC_tier3, EC_tier4 = get_val_counts_EC_tiers(master_df)
    EC_tier1_AS, EC_tier2_AS, EC_tier3_AS, EC_tier4_AS = get_val_counts_EC_tiers(has_AS_df)
    EC_tier1_BS, EC_tier2_BS, EC_tier3_BS, EC_tier4_BS = get_val_counts_EC_tiers(has_BS_df)
    EC_tier1_AS_no_BS, EC_tier2_AS_no_BS, EC_tier3_AS_no_BS, EC_tier4_AS_no_BS = get_val_counts_EC_tiers(has_AS_no_BS_df)
    EC_tier1_BS_no_AS, EC_tier2_BS_no_AS, EC_tier3_BS_no_AS, EC_tier4_BS_no_AS = get_val_counts_EC_tiers(has_BS_no_AS_df)
    
    #plot the distributions of tier1
    plot_2_distributions(EC_tier1_AS, EC_tier1_BS_no_AS, out_path + "tier1AS_vs_BSsinAS.png")
    plot_4_distributions(EC_tier1_AS, EC_tier1_BS, EC_tier1_BS_no_AS, EC_tier1_AS_no_BS, out_path + "all_tier1.png")
    plot_2_distributions(EC_tier2_AS, EC_tier2_BS_no_AS, out_path + "tier2AS_vs_BSsinAS.png")
    plot_2_distributions(EC_tier3_AS, EC_tier3_BS_no_AS, out_path + "tier3AS_vs_BSsinAS.png")

    print(len(has_BS_df))
    
    # save the dfs
    has_BS_df.to_csv('/scratch/project/squid/OMEGA/dfs/has_BS_df.tsv', sep='\t')
    has_BS_no_AS_df.to_csv('/scratch/project/squid/OMEGA/dfs/has_BS_no_AS_df.tsv', sep='\t')
    has_AS_no_BS_df.to_csv('/scratch/project/squid/OMEGA/dfs/has_AS_no_BS_df.tsv', sep='\t')
    has_AS_df.to_csv('/scratch/project/squid/OMEGA/dfs/has_AS_df.tsv', sep='\t')
    master_df.to_csv('/scratch/project/squid/OMEGA/dfs/master_df.tsv', sep='\t')

    
            
                    
            
                        


if __name__ == '__main__':
    main()