"""Ship representation based on static AIS data"""


class Ship:

    """Represents a ship sending AIS data"""

    def __init__(self, sourcemmsi, shipname, shiptype, length, width):
        """Constructor of a Ship

        Parameters:
            sourcemmsi: Identifier of the ship
            shipname: Name of the ship (string)
            shiptype: Type of the ship (integer)
            length: Length of the ship (meter)
            width: Width of the ship (meter)
        """
        self.sourcemmsi = sourcemmsi
        self.shipname = shipname
        self.shiptype = shiptype
        self.shiptypearray = []
        self.length = length
        self.normalized_length = 0.0
        self.width = width
        self.normalized_width = 0.0
        self.ais_states = []
        self.routes = []
