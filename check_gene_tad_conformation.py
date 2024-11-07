import os
import re
from tqdm import tqdm
import time

import multiprocessing
import pandas as pd
import numpy as np
from itertools import combinations

from scipy.optimize import curve_fit
import bioframe as bf

import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

sns.set_style("whitegrid")
sns.set_context("paper")

alignment_code_to_species_legend = {
    'mr': 'Mouse/Rat',
    'cs': 'Cow/Sheep',
    'hr': 'Human/Rhesus',
    'cd': 'Cat/Dog',
    'csp': 'Pig/Cow/Sheep',
    'cspcd': 'Pig/Cow/Sheep/Dog/Cat',
    'mrr': 'Mouse/Rabbit/Rat',
    'hrmrr': 'Human/Mouse/Rhesus/Rabbit/Rat',
    'hrmrrcspcd': 'Mammals',
    'hrmrrcspcdc': 'Mammals/Chicken',
    'hrmrrcspcdcz': 'Mammals/Chicken/Zebrafish'
}
colors = {'mr': "#033270",
          'cs': "#1368AA",
          'hr': "#4091C9",
          'cd': "#4AA8CD",
          'csp': "#BA746B",
          'cspcd': "#F29479",
          'mrr': "#F26A4F",
          'hrmrr': "#EF3C2D",
          'hrmrrcspcd': "#CB1B16",
          'hrmrrcspcdc': "#980E11",
          'hrmrrcspcdcz': "#65010C"
          }
colors_sorted = {i:colors[i] for i in colors.keys()}

div_time = {'mr': 13.10, 'cs': 24.60,'hr': 28.82, 'cd': 55.4, 'csp': 61.8,'cspcd': 76.0,'mrr': 79.0,
            'hrmrr': 87.2, 'hrmrrcspcd': 94.0, 'hrmrrcspcdc': 318.9, 'hrmrrcspcdcz': 429.0}
# div_time = {'mr': 13.10}

def get_code_conversion(code):
    alignment_code_to_species = {'hr':          'human;rhesus',
                                 'mr':          'mouse;rat',
                                 'cd':          'cat;dog',
                                 'cs':          'cow;sheep',
                                 'csp':         'pig;cow;sheep',
                                 'mrr':         'mouse;rabbit;rat',
                                 'cspcd':       'pig;cow;sheep;dog;cat',
                                 'hrmrr':       'human;mouse;rhesus;rabbit;rat',
                                 'hrmrrcspcd':  'human;mouse;rhesus;rabbit;rat;pig;cow;sheep;dog;cat',
                                 'hrmrrcspcdc': 'human;mouse;rhesus;rabbit;rat;pig;cow;sheep;dog;cat;chicken',
                                 'hrmrrcspcdcz':'human;mouse;rhesus;rabbit;rat;pig;cow;sheep;dog;cat;chicken;zfish'
                                 }
    return alignment_code_to_species.get(code)

def read_coord_file(fname):
    with open(fname, 'r') as coord:
        coord_file = coord.readlines()[1:]
        coord_dict = {
            'gene_name': [i.strip().split()[0] for i in coord_file if len(i.strip().split()) == 5],
            'chrom': [i.strip().split()[1] for i in coord_file if len(i.strip().split()) == 5],
            'start': [int(i.strip().split()[2]) for i in coord_file if len(i.strip().split()) == 5],
            'end': [int(i.strip().split()[3]) for i in coord_file if len(i.strip().split()) == 5],
            'strand': [i.strip().split()[4] for i in coord_file if len(i.strip().split()) == 5]}
    coord.close()
    return coord_dict

def get_sb_len(species, aln, path="/Users/fabianpa/Desktop/new_sequences/synteny_3/synteny_bed"):
    sb_dfs = []
    for common_name in species:
        fname = os.path.join(path, common_name + "_" + aln + ".bed")
        sb_df = pd.read_table(fname, names=("chrom", "start", "end", "aln", "strand"), header=None)
        sb_df["species"] = common_name
        sb_df["sb_length"] = sb_df["end"] - sb_df["start"]
        sb_dfs.append(sb_df)
    return pd.concat(sb_dfs, axis=0)

def get_tad_in_sb(species, aln, sb_coord, sb_specific_aln=None, shuffle=False,
                  coord_path="/Users/fabianpa/Desktop/new_sequences/sb_in_tad_3/"):
    count_df = []
    coord_df = []
    for common_name in species:
        coord_fname = os.path.join(coord_path, f"{common_name.lower()}_{aln}.tad.sb")
        # coord_fname = os.path.join(coord_path, f"{common_name.lower()}_{aln}.tad.sb.bed")
        specie_aln_name = common_name.lower() + "_" + aln
        try:
            sb_tad_coord = \
            pd.read_csv(coord_fname, sep="\t", usecols=[5, 6, 7, 3, 4], names=("aln", "strand", "chrom", "start", "end"),
                        header=None)[["chrom", "start", "end", "aln", "strand"]]
            sb_tad_coord["species"] = common_name
            sb_tad_coord["tad_size"] = sb_tad_coord["end"] - sb_tad_coord["start"]
            if sb_specific_aln:
                coord_df.append(sb_tad_coord[sb_tad_coord.aln.isin(sb_specific_aln[specie_aln_name])])
            else:
                coord_df.append(sb_tad_coord)
            count_list = []
            for i in sb_coord.aln.unique():
                count = len(sb_tad_coord[(sb_tad_coord.aln == i)])
                count_list.append([i, count, common_name])

            if sb_specific_aln:
                df = pd.DataFrame(count_list, columns=['aln', 'count', "species"])
                if shuffle: df["count"] = df["count"].sample(frac=1).values
                count_df.append(df[df["aln"].isin(sb_specific_aln[specie_aln_name])])
            else:
                count_df.append(pd.DataFrame(count_list, columns=['aln', 'count', "species"]))

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e.filename}")
    return count_df, coord_df

def get_cynteny_df(aln_name, aln_path='/Users/fabianpa/Desktop/new_sequences/synteny_3/'):
    aln_fname = os.path.join(aln_path, aln_name + ".synteny")
    with open(aln_fname, 'r') as synteny_file:
        synteny = synteny_file.read()
        species = [s.split(".coord")[0].split('/')[-1] for s in synteny.split('\n', 1)[0].split()[1:]]
        found_aln = re.findall(r"Alignment*", synteny, re.MULTILINE)
        n_aln = list(map(lambda x, y: x + str(y), found_aln, range(1, len(found_aln) + 1)))
        all_aln = {key: {} for key in n_aln}

        n_keys = list(range(len(synteny.split('Ali')[1].split('\n')[1].split())))
        for i in synteny.split('Ali'):
            aln = None
            for j in i.split('\n'):
                if j.startswith('gnment '):
                    aln = 'Alignment' + j.split()[1]
                    all_aln[aln] = {key: [] for key in n_keys}
                elif aln:
                    [all_aln[aln][idx].append(j.strip().split()[idx]) for idx in range(len(j.strip().split()))]
    return all_aln, species

def get_equal_tad_count_syn_blocks(tad_count, species):
    s1_count = tad_count.loc[(tad_count.species == species[0])]
    s2_count = tad_count.loc[(tad_count.species == species[1])]

    count_diff = s1_count["count"] - s2_count["count"]
    count_diff_df = pd.DataFrame({"aln": s1_count.aln, "count_s1": s1_count["count"],
                                  "count_s2": s2_count["count"], "count_diff": count_diff})

    return count_diff_df.loc[
        (count_diff_df["count_diff"] == 0) & (count_diff_df["count_s1"] != 0), "aln"].values.tolist()

def get_gene_similarity(tads, alignments, species, gene_coords, shuffle=False):
    cons_score = []

    def prepare_gene_coords(gene_coords_df, species_name, gene_inters):
        coords = pd.DataFrame(gene_coords[species_name])
        coords['gene_name'] = coords['gene_name'].str.split('_').str[0]
        coords = coords[coords['gene_name'].isin(gene_inters)]

        coords.loc[coords['strand'] == '+', 'end'] = coords['start'] + 1
        coords.loc[coords['strand'] == '-', 'start'] = coords['end']
        coords.loc[coords['strand'] == '-', 'end'] = coords['end'] + 1

        coords['chrom'] = 'chr' + coords['chrom'].astype(str)
        return coords

    def prepare_tads_for_species(tads, species_name, aln_number):
        strand = tads.loc[(tads.species == species_name) & (tads.aln == aln_number), "strand"].values[0]
        tad_df = tads.loc[(tads.species == species_name) & (tads.aln == aln_number)].sort_values(
            by=['start'], ascending=(strand != "-")).reset_index(drop=True)
        tad_df['tad_number'] = tad_df.index + 1
        return tad_df

    for aln_number in tads.aln.unique():
        score = 0
        aln = alignments[aln_number].dropna()
        gene_s1 = list(aln[species[0]])
        gene_s2 = list(aln[species[1]])
        # Find intersecting genes between the two species
        gene_inters = set(aln[species[0]].str.split('_').str[0]).intersection(
            set(aln[species[1]].str.split('_').str[0]))
        gene_inters.discard('-')
        if gene_inters:
            # Prepare gene coordinates and TADs for both species
            gene_coords_s1 = prepare_gene_coords(gene_coords, species[0], gene_inters)
            gene_coords_s2 = prepare_gene_coords(gene_coords, species[1], gene_inters)

            tad_s1 = prepare_tads_for_species(tads, species[0], aln_number)
            tad_s2 = prepare_tads_for_species(tads, species[1], aln_number)

            gene_tad_overlap_s1 = bf.overlap(tad_s1, gene_coords_s1, return_overlap=True).dropna()
            gene_tad_overlap_s2 = bf.overlap(tad_s2, gene_coords_s2, return_overlap=True).dropna()

            # Prepare overlap data for merging by ensuring consistent column types
            gene_tad_overlap_s1['gene_name_'] = gene_tad_overlap_s1['gene_name_'].astype(str)
            gene_tad_overlap_s1['tad_number'] = gene_tad_overlap_s1['tad_number'].astype(str)

            gene_tad_overlap_s2['gene_name_'] = gene_tad_overlap_s2['gene_name_'].astype(str)
            gene_tad_overlap_s2['tad_number'] = gene_tad_overlap_s2['tad_number'].astype(str)

            merged_df = pd.merge(gene_tad_overlap_s1, gene_tad_overlap_s2,
                                 on=['gene_name_', 'tad_number'], how='inner')
            score = len(merged_df) / len(gene_inters)
            cons_score.append(score)
    return np.mean(cons_score)

def plot_scatter_fitted(gene_tad_conf_df):
    def power_law(x, a, b):
        return a * np.power(x, b)

    x_data = gene_tad_conf_df["mya"]
    y_data = gene_tad_conf_df["gene_tad_conformation_ratio"]

    plt.figure(figsize=(15, 11))

    # params, covariance = curve_fit(power_law, x_data[:], y_data[:])
    params, covariance = curve_fit(power_law, x_data[3:], y_data[3:])
    a, b = params

    # Generate the fitted curve and confidence interval curve
    # x_fit = np.linspace(min(x_data[:]), max(x_data[:]), 100)
    x_fit = np.linspace(min(x_data[3:]), max(x_data[3:]), 100)
    y_fit = power_law(x_fit, a, b)

    plt.axhline(y=1, color='r', linestyle="--", label='Reference Line')

    sns.lineplot(x=x_fit, y=y_fit, color='grey', linestyle="-.", label='Fitted Curve',)

    # Scatter plot for data points
    sns.scatterplot(x=gene_tad_conf_df["mya"], y=gene_tad_conf_df["gene_tad_conformation_ratio"], hue=gene_tad_conf_df.alignment,
                    palette=list(colors_sorted.values()), s=90)

    # Add labels and legend
    plt.xlabel("Time (Million Years Ago)", fontsize=18)
    plt.ylabel("Gene-TAD vs. Gene-Shuffled TAD Conformation Similarity Ratio", fontsize=18)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [
        f"{label}:{alignment_code_to_species_legend[label]}" if label in alignment_code_to_species_legend.keys() else label
        for label in labels]
    plt.legend(handles=handles[:], labels=labels[:], bbox_to_anchor=(1.5, 0.6), borderaxespad=0, frameon=False,
               fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.savefig("../images/figure 5/edit_ops_fitted.1.png", bbox_inches='tight')
    plt.show()

def main():
    outer_pbar = tqdm(div_time.keys(), position=0, leave=True, colour='blue')
    actual_score = []
    rdm_score = []

    for aln in outer_pbar:
        coord_path = "/Users/fabianpa/Desktop/new_sequences/coord"
        all_aln, species = get_cynteny_df(aln)
        reform = {(outerKey, innerKey): values for outerKey, innerDict in all_aln.items() for innerKey, values in
                  innerDict.items()}
        df = pd.DataFrame.from_dict(reform, orient='index').transpose()
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        column_n = list(range(1, len(df['Alignment1'].columns), 2))
        columns_name = {column_n[i]: species[i] for i in range(len(column_n))}
        df.rename(columns=columns_name, inplace=True)

        sb_coord = get_sb_len(species, aln)
        species_coords = {f: read_coord_file(os.path.join(coord_path, f + ".coord")) for f in species}

        tad_sb_list, tad_sb_coord_list = get_tad_in_sb(species, aln, sb_coord)
        tad_sb_df = pd.concat(tad_sb_list, axis=0)
        tad_sb_coord_df = pd.concat(tad_sb_coord_list, axis=0)

        comb = list(combinations(species, 2))
        score = []
        for i in comb:
        # for i in tqdm(comb, leave=False,colour="green"):
            syn_block = get_equal_tad_count_syn_blocks(tad_sb_df, i)
            comb_df = tad_sb_coord_df.loc[(tad_sb_coord_df.aln.isin(syn_block)) &
                                          ((tad_sb_coord_df.species == i[0]) |
                                           (tad_sb_coord_df.species == i[1]))]
            gene_content_similarity = get_gene_similarity(comb_df, df, i, species_coords)
            score.append(gene_content_similarity)

        actual_score.append([aln, div_time[aln],np.mean(score)])

    gene_tad_conf_df = pd.DataFrame(actual_score, columns=['alignment', 'mya', 'gene_tad_conformation'])
    print(gene_tad_conf_df)

    # rdm_gene_tad_conf_df = pd.DataFrame(rdm_results, columns=['alignment', 'mya', 'gene_tad_conformation'])

    # print(tabulate(observed_score, headers='keys', tablefmt='psql'))
    # print(tabulate(shuffled_score, headers='keys', tablefmt='psql'))

    # merged_df = gene_tad_conf_df.merge(rdm_gene_tad_conf_df, on=['alignment', 'mya'],
    #                                       suffixes=('_actual', '_random'))
    # merged_df['gene_tad_conformation_ratio'] = merged_df['gene_tad_conformation_actual'] / merged_df['gene_tad_conformation_random']
    # merged_df['ratio'] = round(merged_df['gene_tad_conformation_actual'] / merged_df['gene_tad_conformation_random'],1)
    # X = merged_df[['alignment', 'ratio']]
    # print(tabulate(X, headers='keys', tablefmt='psql'))
    # plot_scatter_fitted(merged_df)

if __name__ == "__main__":
    main()