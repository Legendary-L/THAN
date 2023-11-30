with open('data/dblp/oriData/author_label_new.txt', 'r') as f_in, open('data/dblp/oriData/author_label_new3.txt', 'w') as f_out:
    line_index = 0
    for line in f_in:
        index, label = line.strip().split('\t')
        index = int(index)
        while line_index < index:
            f_out.write('{}\t4\n'.format(line_index))
            line_index += 1
        f_out.write('{}\t{}\n'.format(index, label))
        line_index += 1
    while line_index <= 14474:
        f_out.write('{}\t4\n'.format(line_index))
        line_index += 1
