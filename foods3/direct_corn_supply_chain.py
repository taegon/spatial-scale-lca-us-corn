import csv
import os
import numpy as np
from foods3 import util
from gurobipy import *

county_size = 3109


def optimize_gurobi(supply_code, supply_corn, demand_code, demand_corn, dist_mat):
    env = Env("gurobi_spatial_lca.log")

    model = Model("lp_for_spatiallca")
    var = []

    # add constraint for corn product
    # all flow value bigger than equals 0

    no_of_supply = len(supply_code)
    no_of_demand = len(demand_code)

    var = []
    sol = np.zeros(no_of_supply * no_of_demand)

    for i, vs in enumerate(supply_code):
        for j, vd in enumerate(demand_code):
            var.append(model.addVar(0.0, min(supply_corn[i], demand_corn[j]), 0.0, GRB.CONTINUOUS, "S_s[{:d},{:d}]".format(i, j)))
    model.update()
    print("corn flow constraint = all number positive")

    # Set objective: minimize cost
    expr = LinExpr()

    for i, vs in enumerate(supply_code):
        for j, vd in enumerate(demand_code):
            expr.addTerms(dist_mat[i][j], var[i * no_of_demand + j])

    model.setObjective(expr, GRB.MINIMIZE)

    # sum of supply(specific row's all columns) is small than product of corn
    # Add constraint
    for i, vs in enumerate(supply_code):
        expr = LinExpr()
        for j, vd in enumerate(demand_code):
            expr.addTerms(1.0, var[i * no_of_demand + j])
        model.addConstr(expr, GRB.LESS_EQUAL, supply_corn[i], "c{:d}".format(i + 1))

    print("sum of corn flow from specific county smaller than total product of that county")

    # sum of supply (specific column's all row) is equals to the demand of county
    for j, vd in enumerate(demand_code):
        expr = LinExpr()
        for i, vs in enumerate(supply_code):
            expr.addTerms(1.0, var[i * no_of_demand + j])
        model.addConstr(expr, GRB.EQUAL, demand_corn[j], "d{:d}".format(j + 1))

    print("all constraints are set.")

    # Optimize model
    model.optimize()

    for i, vs in enumerate(supply_code):
        for j, vd in enumerate(demand_code):
            sol[i * no_of_demand + j] = var[i * no_of_demand + j].x

    return sol


def read_csv_int(filename, col_idx):
    values = []
    with open(filename, "r", encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            v = row[col_idx]
            values.append(int(v))
    return values


def read_csv_float(filename, col_idx):
    values = []
    with open(filename, "r", encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            v = row[col_idx]
            v = v.replace(",", "")
            # print(v)
            if v is None or v == "" or v.strip() == "-":
                values.append(0)
            else:
                values.append(float(v))
    return values


def read_csv_float_range(filename, col_idx, col_idx_end):
    values = []
    with open(filename, "r", encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            sum_value = 0.
            for col in range(col_idx, col_idx_end):
                v = row[col]
                v = v.replace(",", "")
                if v is None or v == "" or v.strip() == "-":
                    v = 0
                else:
                    v = float(v)
                sum_value += v
            values.append(sum_value)
    return values


def read_dist_matrix(filename):
    matrix = np.zeros((county_size, county_size))
    with open(filename, "r") as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            for c in range(county_size):
                matrix[i][c] = float(row[c])
    return matrix


def expand_list(corn_demand_file, input_file, output_file):
    demand = {}
    with open(corn_demand_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            demand[row[0]] = [float(row[8]), float(row[9]), float(row[10]),
                              float(row[11]), float(row[12]), float(row[13]),
                              float(row[7])]

        sub_sector = ["layer", "pullet", "turkey", "milkcow", "wetmill", "export", "others"]

    data_list = []
    with open(input_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data_list.append(row)

    expanded_list = []
    for row in data_list:
        if row[0] == "others":
            weighted_col_idx = [3,]
            target_county = row[1]
            total_demand = sum(demand[target_county])
            for ss in range(len(sub_sector)):
                if total_demand == 0:
                    weight = 1
                else:
                    weight = demand[target_county][ss] / total_demand
                split_row = [row[x] if x not in weighted_col_idx else float(row[x])*weight for x in range(len(row))]
                split_row[0] = sub_sector[ss]
                if split_row[3] != 0:
                    expanded_list.append(split_row)
        else:
            expanded_list.append(row)

    with open(output_file, "w") as f:
        f.write(",".join(header))
        f.write("\n")
        for row in expanded_list:
            f.write(",".join([str(x) for x in row]))
            f.write("\n")


def main(output_filename, demand_filename):
    county_code = read_csv_int("../input/county_FIPS.csv", 0)

    supply_code = county_code[:]
    supply_amount = read_csv_float(demand_filename, 1)

    demand_code = []
    for i in range(5):
        demand_code.extend(county_code)

    demand_amount = []
    # cattle(0), poultry(1), ethanol(2), hog(3), others(4)
    demand_amount.extend(read_csv_float(demand_filename, 3))
    demand_amount.extend(read_csv_float(demand_filename, 5))
    demand_amount.extend(read_csv_float(demand_filename, 6))
    demand_amount.extend(read_csv_float(demand_filename, 4))
    demand_amount.extend(read_csv_float_range(demand_filename, 7, 14))

    print(sum(supply_amount))
    print(sum(demand_amount))

    all_imp_filename = "../input/allDist_imp.csv"
    dist_imp_all_matrix = read_dist_matrix(all_imp_filename)

    dist_mat = np.zeros((len(supply_code), len(demand_code)))

    print("making distance matrix")
    dist_mat[0:3109, 0 + 0 * 3109:3109 * 1] = dist_imp_all_matrix
    dist_mat[0:3109, 0 + 1 * 3109:3109 * 2] = dist_imp_all_matrix
    dist_mat[0:3109, 0 + 2 * 3109:3109 * 3] = dist_imp_all_matrix
    dist_mat[0:3109, 0 + 3 * 3109:3109 * 4] = dist_imp_all_matrix
    dist_mat[0:3109, 0 + 4 * 3109:3109 * 5] = dist_imp_all_matrix

    print("run simulation model")
    sol = optimize_gurobi(supply_code, supply_amount, demand_code, demand_amount, dist_mat)

    no_of_demand = len(demand_code)
    sector_name = ("cattle", "broiler", "ethanol", "hog", "others")

    with open(output_filename, "w") as f:
        headline = [
            "sector", "demand_county", "corn_county", "corn_bu",
        ]
        f.write(",".join(headline))
        f.write("\n")
        for i, v in enumerate(sol):
            if v > 0:
                sector = (i % no_of_demand) // county_size
                src_county_idx = i // no_of_demand
                des_county_idx = i % no_of_demand % county_size
                supply_corn_bu = v
                src_county_fips = county_code[src_county_idx]
                des_county_fips = county_code[des_county_idx]
                f.write("{},{},{},{}\n".format(sector_name[sector], des_county_fips, src_county_fips, supply_corn_bu))


if __name__ == '__main__':
    ROOT_DIR = util.get_project_root()
    output_dir = ROOT_DIR / "output"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    corn_flow_filename = "../output/corn_flow_county_scale_major_category.csv"
    corn_demand_filename = "../input/corn_demand_2012.csv"
    main(corn_flow_filename, corn_demand_filename)
    expand_list(corn_demand_filename,
                corn_flow_filename,
                "../output/impacts_scale_county_all_category.csv")
