module RobustNMFExampleDatasets

using DataDeps


"""
    register_att_faces!()

Registers the AT&T / ORL face dataset with DataDeps.jl under the name "ATT_FACES_ORL".

Data source:
- University of Cambridge AT&T archive (att_faces.zip)
"""
function register_att_faces!()
    name = "ATT_FACES_ORL"

    if haskey(DataDeps.registry, name)
        return
    end

    msg = """
    AT&T / ORL Database of Faces (PGM grayscale faces).

    Source: University of Cambridge AT&T archive.
    Please give appropriate credit to AT&T Laboratories Cambridge when using this dataset.

    This dataset will be downloaded and extracted locally by DataDeps.jl
    """

    url = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"

    DataDeps.register(DataDep(
        name,
        msg,
        url;
        post_fetch_method = DataDeps.unpack
    ))
end


"""
    att_faces_root() -> String

Returns the local path to the extracted dataset root directory.
"""
function att_faces_root()
    register_att_faces!()
    return datadep"ATT_FACES_ORL"
end

end # module