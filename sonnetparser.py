# -*- coding: utf-8 -*-

import re
import functools
import warnings
import numpy as np


def _parse_sonnet_output_file(lines):
    """Parse a chunk of a Sonnet output file for a single parameter set.

    Can handle files with or without column header.
    """
    assert lines[0].strip() == 'Sonnet Data File'
    # Comments (lines starting with space)
    commentsend = 0
    while lines[commentsend].startswith(' '):
        commentsend += 1
    headerend = commentsend
    # Discard project date
    assert lines[headerend].startswith('Project Date:')
    headerend += 1
    # Parse port termination
    line = lines[headerend].split()
    if line[0] == 'R':
        assert len(line) == 2
        termination = float(line[1])
    elif line[0] == 'TERM':
        # List of resistance, reactance for each port
        assert (len(line)-1) % 2 == 0
        nports = (len(line)-1) // 2
        termination = [(float(line[i+1]), float(line[i+2])) for i in range(nports)]
    else:
        raise ValueError(f"Unknow termination format in line {headerend}: "+lines[headerend])
    headerend += 1
    # Optional column header
    if lines[headerend].startswith('Frequency'):
        columns = [c.strip() for c in lines[headerend].split(',')]
        headerend += 1
    else:
        columns = None
    # Parameter list
    paramend = headerend
    while '=' in lines[paramend]:
        paramend += 1
    params = [l.split('=') for l in lines[headerend:paramend]]
    params = {n.strip(): float(v) for n, v in params}
    # Parse data
    data = np.array([
        [float(v.strip()) for v in l.split(',')]
        for l in lines[paramend:] if l
    ])
    return {
        'comments': lines[:commentsend],
        'termination': termination,
        'columns': columns,
        'params': params,
        'data': data
    }


def load_sonnet_output_csv(fname):
    """Load file generated as simulation output file in 'Spreadsheet' format.

    Returns a list of chunks each being a dict with information
    for one parameter value set.
    """
    with open(fname) as f:
        lines = f.readlines()
    chunks = []
    while lines:
        try:
            nextdata = lines[1:].index(' Sonnet Data File\n')
            chunk, lines = lines[:nextdata+1], lines[nextdata+1:]
        except ValueError:
            # String not found, last chunk
            chunk, lines = lines, None
        chunks.append(_parse_sonnet_output_file(chunk))
    return chunks


def load_sonnet_output_Sparam(fname, pnames=[], columnfmt=None):
    """Load Sonnet output S parameter into one large numpy array.

    `pnames` can be a list of parameter names to determine the parameters to be
    tabulated if not all parameters should become axes of the output.

    `columnfmt` is a hint for the column format, needed if no column header
    is present in the file. Use `'MAG-ANG'`, `'DB-ANG'`, or `'RE-IM'`.

    Leading axes of the array are the parameter values, then the frequency,
    then the S matrix axes. I.e. a project with two parameters with 10 values each,
    300 frequency points and 4 ports will result in an array of shape (10, 10, 300, 4, 4).

    Will use all frequency points for any parameter value. Frequency points missing
    for other parameter values are left NaN.

    Outputs a 3-tuple of parameter values, frequency points, and the data:
    {pname: [values], pname2: [values], ...},
    frequency points,
    numpy array of shape (param1, param2, ..., frequencies, nports, nports).
    """
    chunks = load_sonnet_output_csv(fname)

    # # Find maximal set of frequencies that are present in all chunks
    # allfs = [set(chunk['data'][:,0]) for chunk in chunks]
    # fs = allfs[0].intersection(*allfs[1:])
    # assert fs, "No intersection between frequencies saved for different parameters."
    # fs = np.array(sorted(list(fs)))

    # Maximal set of frequencies in any chunk
    fs = functools.reduce(lambda a, b: a | b, [
        set(chunk['data'][:,0]) for chunk in chunks])
    fs = np.array(sorted(list(fs)))

    # Get parameter values
    #pkeys = list(set(itertools.chain(*list(chunk['params'].keys() for chunk in chunks))))
    pvalues = []
    for pname in pnames:
        pvalues.append(sorted(list(set(chunk['params'][pname] for chunk in chunks))))
    pshape = tuple(len(vs) for vs in pvalues)

    assert (chunks[0]['data'].shape[1]-1) % 2 == 0
    nports = int(((chunks[0]['data'].shape[1]-1) / 2)**0.5)
    assert nports**2 * 2 + 1 == chunks[0]['data'].shape[1]
    # Same number of ports in all chunks?!
    assert all(chunk['data'].shape[1] == nports**2 * 2 + 1 for chunk in chunks)

    Sdata = np.full(pshape+(len(fs),nports,nports), np.nan+0j)

    for chunk in chunks:
        dat = chunk['data']
        # Select frequencies, assumes frequencies in table are sorted
        fs2 = dat[:,0]
        # maskright = np.array([f in fs for f in fs2])
        maskleft = np.array([f in fs2 for f in fs])
        # find index in Sdata
        pidx = tuple(vs.index(chunk['params'][n]) for n, vs in zip(pnames, pvalues))

        # Determine column format
        if columnfmt is None:
            if 'columns' in chunk:
                # Assert same format for all following columns
                fmt1 = chunk['columns'][1].partition('[')[0]
                fmt2 = chunk['columns'][2].partition('[')[0]
                fmt = fmt1 + '-' + fmt2
            else:
                raise ValueError("Cannot determine column format because no column header is present. Use columnfmt argument.")
        else:
            fmt = columnfmt
        for i in range(nports):
            for j in range(nports):
                colidx = (i * nports + j) * 2
                if fmt == 'MAG-ANG':
                    assert dat[:,colidx] >= 0
                    assert -180 <= dat[:,colidx] <= 180
                    Sdata[pidx][maskleft,i,j] = dat[:,colidx] * np.exp(1j*dat[:,colidx+1]/180*np.pi)
                elif fmt == 'RE-IM':
                    Sdata[pidx][maskleft,i,j] = dat[:,colidx] + 1j*dat[:,colidx+1]
                else:
                    raise ValueError(f"S parameter format {fmt} not implemented.")

    # Check that all parameter combinations are in dataset
    if np.any(np.isnan(Sdata)):
        warnings.warn("Missing parameter combinations or frequency points. Loaded S matrix array contains NaNs.")

    # Convert parameter values to numpy arrays
    #pvalues = [np.array(v) for v in pvalues]
    pdict = {n: np.array(v) for n, v in zip(pnames, pvalues)}
    return pdict, fs, Sdata


# Matches the line separating chunks of data in the plot output.
#   First group is the file name, assuming .son file extension,
#   Second group is settings or info like 'NET_LIST',
#   Third group is parameters.
# Parameters must not end in .son.
# Parameters and values must have equal sign without spaces between them.
RE_PROJECT_PATH = re.compile(r'(.+\.son)( [^=]+)?( .*)?')

def load_sonnet_graph_csv(fname):
    """Load file generated by saving "Output > All curves to Spreadsheet" in plot.

    Result is a list, each element a dict in the plot with keys:
    path, settings, header,
    params: dict with parameters as keys
    data: 2-N numpy array.

    Doesn't assemble results into array, because each line may have
    different samples, since Sonnet mixes new and cached simulation results.
    """
    with open(fname) as f:
        chunks = []
        for line in f:
            mo = RE_PROJECT_PATH.fullmatch(line.strip())
            if mo:
                paramline = mo.group(3) or ''
                params = [pair.split('=') for pair in paramline.split()]
                params = {n.strip(): float(v) for n, v in params}
                chunks.append({
                    'path': mo.group(1),
                    'settings': mo.group(2).strip().split(),
                    'params': params,
                    'header': next(f).strip(),
                    'data': []
                })
            else:
                values = [float(v) for v in line.split(',')]
                chunks[-1]['lines'].append(values)

    # Convert data to numpy arrays
    for chunk in chunks:
        chunk['data'] = np.array(chunk['data'])
    return chunks
